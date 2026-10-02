"""Check episode slices against actual episode boundaries."""
import unittest
import numpy as np
from diffusion_policy.common.replay_buffer import ReplayBuffer


class ReplayEpisodeSliceTest(unittest.TestCase):
    def setUp(self):
        self.buffer = ReplayBuffer.create_empty_numpy()
        for length in (2, 4, 3):
            begin = self.buffer.n_steps
            self.buffer.add_episode({"action": np.arange(begin, begin + length)[:, None]})

    def test_negative_indices_select_the_same_episode(self):
        for index in range(-3, 3):
            with self.subTest(index=index):
                sl = self.buffer.get_episode_slice(index)
                np.testing.assert_array_equal(
                    self.buffer["action"][sl], self.buffer.get_episode(index)["action"]
                )

    def test_last_episode_excludes_prior_episodes(self):
        sl = self.buffer.get_episode_slice(-1)
        self.assertEqual((sl.start, sl.stop), (6, 9))
        np.testing.assert_array_equal(self.buffer["action"][sl, 0], [6, 7, 8])

    def test_invalid_indices_are_rejected(self):
        for index in (-4, 3):
            with self.subTest(index=index), self.assertRaises(IndexError):
                self.buffer.get_episode_slice(index)


if __name__ == "__main__":
    unittest.main()

