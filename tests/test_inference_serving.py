"""Serving: the page's workload and scheduler, ragged batched decoding, paged memory, prefix cache."""

import statistics

import pytest
import torch
import torch.nn.functional as F

from dmxsf.architecture.configs import small_gpt
from dmxsf.architecture.gpt import GPT
from dmxsf.inference.generate import generate
from dmxsf.inference.kv_cache import KVCache
from dmxsf.inference.serving import (
    BatchedDecoder,
    BlockAllocator,
    PagedKV,
    PrefixCache,
    block_hashes,
    machines_needed,
    make_lengths,
    queue_simulation,
    serve,
    simulate,
    step_from_table,
)

STEP = {
    1: 0.0249,
    2: 0.0404,
    3: 0.0562,
    4: 0.0579,
    6: 0.0822,
    8: 0.0984,
    12: 0.0965,
    16: 0.1003,
    24: 0.1016,
    32: 0.1035,
}
STALL = 0.343


def model(**kw):
    torch.manual_seed(0)
    base = dict(vocab_size=32, n_layer=2, n_head=4, n_kv_head=2, d_model=32, block_size=64)
    return GPT(small_gpt(**{**base, **kw})).eval()


# --- the page's workload and simulator ------------------------------------------------------------


def test_page_workload_and_simulation():
    lengths = make_lengths(48, 7)
    assert (
        min(lengths),
        statistics.median(lengths),
        round(statistics.mean(lengths)),
        max(lengths),
    ) == (
        15,
        61,
        97,
        453,
    )
    assert sum(lengths) == 4656
    step = step_from_table(STEP)
    cont = simulate(lengths, 16, True, STALL, step)
    static = simulate(lengths, 16, False, STALL, step)
    assert round(max(cont), 1) == 52.5 and round(max(static), 1) == 68.6
    assert len(cont) == len(static) == 48
    # identical lengths: the two schedulers have nothing to gain from each other
    same = [100] * 48
    assert round(max(simulate(same, 16, True, STALL, step)), 1) == 46.6
    assert max(simulate(same, 16, False, STALL, step)) == pytest.approx(
        max(simulate(same, 16, True, STALL, step))
    )


def test_step_interpolation():
    step = step_from_table({1: 1.0, 5: 2.0})
    assert step(1) == 1.0 and step(3) == 1.5 and step(5) == 2.0 and step(9) == 2.0


# --- ragged batched decoding ----------------------------------------------------------------------


def test_batched_step_matches_decoding_each_sequence_alone():
    m = model()
    g = torch.Generator().manual_seed(1)
    prompts = [torch.randint(0, 32, (n,), generator=g) for n in (3, 11, 7, 20)]
    dec = BatchedDecoder(m, slots=4, max_len=64)
    solo = [KVCache(2) for _ in prompts]
    last = []
    for s, p in enumerate(prompts):
        a = dec.prefill(s, p)
        b = m(p[None], cache=solo[s])[0, -1]
        assert torch.allclose(a, b, atol=1e-5)
        last.append(int(b.argmax()))
    for _ in range(6):  # sequences of four different lengths advance together
        batched = dec.step([0, 1, 2, 3], torch.tensor(last))
        nxt = []
        for s in range(4):
            alone = m(torch.tensor([[last[s]]]), cache=solo[s])[0, -1]
            assert torch.allclose(batched[s], alone, atol=1e-4)
            nxt.append(int(alone.argmax()))
        last = nxt
    assert dec.length == [9, 17, 13, 26]


def test_a_subset_of_slots_can_step_and_a_freed_slot_can_be_reused():
    m = model()
    p = torch.arange(5)
    dec = BatchedDecoder(m, slots=3, max_len=64)
    for s in range(3):
        dec.prefill(s, p)
    dec.step([2], torch.tensor([1]))
    assert dec.length == [5, 5, 6]
    dec.free(2)
    dec.prefill(2, torch.arange(9))
    assert dec.length == [5, 5, 9]
    full = BatchedDecoder(m, slots=1, max_len=5)
    full.prefill(0, torch.arange(5))
    with pytest.raises(ValueError):  # the slot has no room for one more token
        full.step([0], torch.tensor([1]))


@pytest.mark.parametrize("continuous", [True, False])
def test_serve_gives_every_request_its_own_greedy_text(continuous):
    m = model()
    g = torch.Generator().manual_seed(2)
    reqs = [
        (torch.randint(0, 32, (n,), generator=g), k)
        for n, k in [(4, 5), (9, 1), (6, 12), (3, 3), (8, 7)]
    ]
    out = serve(m, reqs, slots=2, continuous=continuous, max_len=64)
    for (prompt, k), res in zip(reqs, out, strict=True):
        alone = generate(m, prompt[None], k, temperature=0.0)[0].tolist()
        assert res.tokens == alone
        assert 0 <= res.admitted <= res.first_token <= res.done


def test_continuous_batching_refills_slots_static_batching_does_not():
    m = model()
    reqs = [(torch.arange(4), n) for n in (2, 30, 2, 2, 2, 2)]
    cont = serve(m, reqs, slots=2, continuous=True, max_len=64)
    stat = serve(m, reqs, slots=2, continuous=False, max_len=64)
    # the second request holds a slot for 30 steps; the others pass through the other slot at once
    assert cont[5].admitted < cont[1].done
    assert stat[2].admitted >= stat[1].done and stat[4].admitted >= stat[3].done


# --- paged memory -------------------------------------------------------------------------------


def test_block_allocator_counts_references():
    a = BlockAllocator(3, 4)
    b0, b1 = a.allocate(), a.allocate()
    a.share(b0)
    a.release(b0)
    assert a.used == 2
    a.release(b0)
    assert a.used == 1
    a.allocate()
    a.allocate()
    with pytest.raises(MemoryError):
        a.allocate()
    assert b1 in range(3)


def attend(q, k, v):  # q (H, 1, D), k/v (H, T, D)
    return F.scaled_dot_product_attention(q[None], k[None], v[None])[0]


def test_paged_attention_equals_contiguous_attention():
    g = torch.Generator().manual_seed(3)
    kv = PagedKV(n_blocks=12, block_size=4, n_heads=2, head_dim=8)
    seqs = [kv.new_sequence() for _ in range(3)]
    dense = [([], []) for _ in seqs]
    for step in range(
        11
    ):  # tokens of three sequences arrive interleaved, so their blocks interleave
        for i, s in enumerate(seqs):
            if step < 5 + 3 * i:
                k, v = torch.randn(2, 1, 8, generator=g), torch.randn(2, 1, 8, generator=g)
                kv.append(s, k, v)
                dense[i][0].append(k)
                dense[i][1].append(v)
    q = torch.randn(2, 1, 8, generator=g)
    for s, (ks, vs) in zip(seqs, dense, strict=True):
        k, v = kv.gather(s)
        assert torch.equal(k, torch.cat(ks, 1)) and torch.equal(v, torch.cat(vs, 1))
        assert torch.equal(attend(q, k, v), attend(q, torch.cat(ks, 1), torch.cat(vs, 1)))
    assert seqs[0].table != sorted(seqs[0].table) or seqs[0].table[1] - seqs[0].table[0] != 1
    used_before = kv.alloc.used
    for s in seqs:
        kv.free(s)
    assert kv.alloc.used == 0 and used_before == sum(-(-s_len // 4) for s_len in (5, 8, 11))


def test_fork_shares_blocks_and_writes_copy_on_write():
    g = torch.Generator().manual_seed(4)
    kv = PagedKV(n_blocks=10, block_size=4, n_heads=1, head_dim=2)
    parent = kv.new_sequence()
    kv.append(
        parent, torch.randn(1, 6, 2, generator=g), torch.randn(1, 6, 2, generator=g)
    )  # 2 blocks
    before = kv.gather(parent)
    child = kv.fork(parent)
    assert kv.alloc.used == 2  # nothing was copied
    kv.append(child, torch.randn(1, 1, 2, generator=g), torch.randn(1, 1, 2, generator=g))
    assert (
        kv.copies == 1 and kv.alloc.used == 3
    )  # the half-full last block was copied, the first shared
    assert torch.equal(kv.gather(parent)[0], before[0])
    assert child.table[0] == parent.table[0] and child.table[1] != parent.table[1]
    assert torch.equal(kv.gather(child)[0][:, :6], before[0])
    kv.free(child)
    kv.free(parent)
    assert kv.alloc.used == 0


# --- prefix cache ---------------------------------------------------------------------------------


def test_block_hashes_depend_on_everything_in_front():
    a = list(range(40))
    b = [99, *a[1:]]
    c = [*a[:16], 99, *a[17:]]
    ha, hb, hc = (block_hashes(x, 16) for x in (a, b, c))
    assert len(ha) == 2  # only full blocks
    assert ha[0] != hb[0] and ha[1] != hb[1]  # a change in block 0 changes block 1's hash too
    assert ha[0] == hc[0] and ha[1] != hc[1]
    assert block_hashes(a, 16, salt="tenant") != ha  # another salt, another namespace


def test_prefix_cache_reuses_blocks_without_changing_logits():
    m = model(block_size=128)
    g = torch.Generator().manual_seed(5)
    doc = torch.randint(0, 32, (40,), generator=g)
    cache = PrefixCache(m, block_size=8)
    q = [torch.randint(0, 32, (6,), generator=g) for _ in range(3)]
    stats = []
    for i in range(3):
        prompt = torch.cat([doc, q[i]])
        logits, kv, st = cache.prefill(prompt)
        ref = m(prompt[None])[0, -1]
        assert torch.allclose(logits, ref, atol=1e-5)
        assert kv.length == len(prompt)
        stats.append((st.reused, st.computed))
    assert stats == [(0, 46), (40, 6), (40, 6)]
    # a different first token: no block matches
    logits, _, st = cache.prefill(torch.cat([torch.tensor([7]), doc, q[0]]))
    assert st.reused == 0
    # switching the cache off recomputes everything
    assert cache.prefill(torch.cat([doc, q[0]]), use_cache=False)[2].reused == 0
    # tenants with different salts cannot read each other's blocks
    assert cache.prefill(torch.cat([doc, q[0]]), salt="b")[2].reused == 0


def test_prefix_cache_computes_at_least_one_token():
    m = model()
    cache = PrefixCache(m, block_size=8)
    prompt = torch.arange(16)
    cache.prefill(prompt)
    logits, _, st = cache.prefill(
        prompt
    )  # the whole prompt is cached, the last block is recomputed
    assert st.reused == 8 and st.computed == 8
    assert torch.allclose(logits, m(prompt[None])[0, -1], atol=1e-5)


# --- Little's law ---------------------------------------------------------------------------------


def test_littles_law_in_a_simulated_queue():
    rate, w, ell = queue_simulation(rate=3.0, mean_service=1.0, servers=4, n=20000, seed=1)
    assert ell == pytest.approx(rate * w, rel=0.005)
    # with a single, nearly idle server the queue is empty: W is the service time
    _, w1, l1 = queue_simulation(rate=0.05, mean_service=1.0, servers=1, n=20000, seed=2)
    assert w1 == pytest.approx(1.0, abs=0.1) and l1 == pytest.approx(0.05 * w1, rel=0.01)


def test_machines_needed_follows_the_page_example():
    speed = {1: 40.1, 2: 24.8, 3: 17.8, 4: 17.3, 6: 12.2, 8: 10.2}
    ttft = {1: 0.19, 2: 0.31, 3: 0.39, 4: 0.47, 6: 0.76, 8: 0.94}
    per, w, flight, machines = machines_needed(0.8, 200, 17.0, speed, ttft)
    assert per == 4 and w == pytest.approx(0.47 + 199 / 17.3) and round(w, 1) == 12.0
    assert flight == pytest.approx(0.8 * w) and round(flight, 1) == 9.6 and machines == 3
