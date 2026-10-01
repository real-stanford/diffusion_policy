import unittest

import torch

from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


def empirical_stats(data):
    return dict(min=data.amin(0), max=data.amax(0), mean=data.mean(0), std=data.std(0))


class NormalizerOutputStatsTests(unittest.TestCase):
    def setUp(self):
        self.data = torch.tensor([[10., -9.], [20., -3.], [35., 6.], [45., 12.]])

    def assert_stats(self, actual, data):
        for key, expected in empirical_stats(data).items():
            with self.subTest(stat=key):
                torch.testing.assert_close(actual[key], expected)

    def test_single_field_moments_match_normalized_samples(self):
        for mode in ('limits', 'gaussian'):
            for fit_offset in (False, True):
                with self.subTest(mode=mode, fit_offset=fit_offset):
                    normalizer = SingleFieldLinearNormalizer.create_fit(self.data, mode=mode, fit_offset=fit_offset)
                    self.assert_stats(normalizer.get_output_stats(), normalizer.normalize(self.data))

    def test_container_default_and_named_fields_share_statistics(self):
        default = LinearNormalizer()
        default.fit(self.data, mode='gaussian')
        self.assert_stats(default.get_output_stats(), default.normalize(self.data))
        grouped = LinearNormalizer()
        grouped.fit({'obs': self.data, 'action': self.data * 2. + 3.}, mode='limits')
        outputs = grouped.normalize({'obs': self.data, 'action': self.data * 2. + 3.})
        for key, stats in grouped.get_output_stats().items():
            self.assert_stats(stats, outputs[key])
            self.assert_stats(grouped[key].get_output_stats(), outputs[key])

    def test_manual_negative_scales_swap_extrema_and_keep_std_nonnegative(self):
        normalizer = SingleFieldLinearNormalizer.create_manual(
            scale=torch.tensor([-2., 0.5]), offset=torch.tensor([7., -4.]),
            input_stats_dict=empirical_stats(self.data),
        )
        self.assert_stats(normalizer.get_output_stats(), normalizer.normalize(self.data))

    def test_round_trip_state_preserves_output_moments(self):
        source = LinearNormalizer()
        source.fit({'obs': self.data}, mode='gaussian')
        restored = LinearNormalizer()
        restored.load_state_dict(source.state_dict())
        self.assert_stats(restored.get_output_stats()['obs'], restored.normalize({'obs': self.data})['obs'])


if __name__ == "__main__":
    unittest.main()
