import math
import os
import sys
import unittest

import numpy as np
import paddle

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# The fastdeploy.model_executor.ops package eagerly imports optional backend modules
# whose dependencies may be unavailable in the CI environment. Pre-register a lightweight
# stub for the iluvatar backend so that the import of the GPU module does not fail when
# those optional dependencies are missing.
if "fastdeploy.model_executor.ops.iluvatar" not in sys.modules:
    import types

    sys.modules["fastdeploy.model_executor.ops.iluvatar"] = types.ModuleType(
        "fastdeploy.model_executor.ops.iluvatar"
    )

try:  # pragma: no branch - best effort import guarded by runtime availability
    from fastdeploy.model_executor.ops.gpu import multi_head_latent_attention
except (ImportError, AttributeError):
    multi_head_latent_attention = None


def _is_cuda_available() -> bool:
    return paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0

CUDA_AVAILABLE = _is_cuda_available()
OP_AVAILABLE = multi_head_latent_attention is not None


@unittest.skipUnless(
    CUDA_AVAILABLE and OP_AVAILABLE,
    "multi_head_latent_attention requires CUDA and the compiled GPU custom op.",
)
class TestMultiHeadLatentAttention(unittest.TestCase):
    def setUp(self):
        os.environ.setdefault("FLAGS_mla_use_tensorcore", "0")
        self.place = paddle.CUDAPlace(0)

        self.batch_size = 1
        self.token_num = 1
        self.q_num_heads = 1
        self.kv_num_heads = 1
        self.head_dim_qk = 8
        self.head_dim_v = 8
        self.block_size = 64
        self.max_blocks_per_seq = 1
        self.max_seq_len = 64
        self.softmax_scale = 1.0 / math.sqrt(self.head_dim_qk)

    def _build_inputs(self, dtype: str, max_dec_len: int = 0):
        query = paddle.zeros(
            [self.token_num, self.q_num_heads * self.head_dim_qk], dtype=dtype, place=self.place
        )
        key_cache = paddle.zeros(
            [self.max_blocks_per_seq, self.kv_num_heads, self.block_size, self.head_dim_qk],
            dtype=dtype,
            place=self.place,
        )
        value_cache = paddle.zeros(
            [self.max_blocks_per_seq, self.kv_num_heads, self.block_size, self.head_dim_v],
            dtype=dtype,
            place=self.place,
        )

        seq_lens_decoder = paddle.zeros([self.batch_size], dtype="int32", place=self.place)
        seq_lens_this_time = paddle.full([self.batch_size], self.token_num, dtype="int32", place=self.place)
        cu_seqlens_q = paddle.to_tensor([0, self.token_num], dtype="int32", place=self.place)
        batch_id_per_token = paddle.zeros([self.token_num], dtype="int32", place=self.place)
        block_tables = paddle.zeros(
            [self.batch_size, self.max_blocks_per_seq], dtype="int32", place=self.place
        )

        kv_batch_ids = paddle.zeros([self.max_blocks_per_seq], dtype="int32", place=self.place)
        kv_tile_ids_per_batch = paddle.zeros([self.max_blocks_per_seq], dtype="int32", place=self.place)
        kv_num_blocks = paddle.full([1], self.max_blocks_per_seq, dtype="int32", place=self.place)
        decoder_batch_ids = paddle.zeros([1], dtype="int32", place=self.place)
        decoder_tile_ids_per_batch = paddle.zeros([1], dtype="int32", place=self.place)
        decoder_num_blocks = paddle.full([1], self.max_blocks_per_seq, dtype="int32", place=self.place)
        decoder_chunk_size_device = paddle.full([1], self.block_size, dtype="int32", place=self.place)
        max_dec_len_this_time = paddle.full([1], max_dec_len, dtype="int32", place=self.place)
        max_len_kv = paddle.full([1], max_dec_len, dtype="int32", place=self.place)

        compute_dtype = "bfloat16" if dtype == "bfloat16" else "float16" if dtype == "float16" else dtype

        return [
            query,
            key_cache,
            value_cache,
            seq_lens_decoder,
            seq_lens_this_time,
            cu_seqlens_q,
            batch_id_per_token,
            block_tables,
            kv_batch_ids,
            kv_tile_ids_per_batch,
            kv_num_blocks,
            decoder_batch_ids,
            decoder_tile_ids_per_batch,
            decoder_num_blocks,
            decoder_chunk_size_device,
            max_dec_len_this_time,
            max_len_kv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            compute_dtype,
            "none",
            self.head_dim_v,
            self.max_seq_len,
            self.softmax_scale,
            0.0,
            0.0,
            0.0,
            0,
            True,
            False,
        ]

    def test_zero_decode_tokens_returns_zero_output(self):
        for dtype in ("float16", "bfloat16"):
            with self.subTest(dtype=dtype):
                args = self._build_inputs(dtype, max_dec_len=0)
                out = multi_head_latent_attention(*args)

                expected_shape = [self.token_num, self.q_num_heads * self.head_dim_v]
                self.assertEqual(list(out.shape), expected_shape)

                np_out = out.astype("float32").cpu().numpy()
                np.testing.assert_array_equal(np_out, np.zeros_like(np_out))

    def test_unsupported_dtype_raises(self):
        args = self._build_inputs("float32", max_dec_len=0)
        with self.assertRaisesRegex(RuntimeError, "Only float16 and bfloat16"):
            multi_head_latent_attention(*args)


if __name__ == "__main__":
    unittest.main()
