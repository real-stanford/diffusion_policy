import gym
import numpy as np
from diffusion_policy.real_world.video_recorder import VideoRecorder

class VideoRecordingWrapper(gym.Wrapper):
    def __init__(self, 
            env, 
            video_recoder: VideoRecorder,
            mode='rgb_array',
            file_path=None,
            steps_per_render=1,
            **kwargs
        ):
        """
        When file_path is None, don't record.
        """
        super().__init__(env)
        
        self.mode = mode
        self.render_kwargs = kwargs
        self.steps_per_render = steps_per_render
        self.file_path = file_path
        self.video_recoder = video_recoder

        self.step_count = 0

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.frames = list()
        self.step_count = 1
        self.video_recoder.stop()
        return obs
    
    def step(self, action):
        result = super().step(action)
        self.step_count += 1
        if self.file_path is not None \
            and ((self.step_count % self.steps_per_render) == 0):
            if not self.video_recoder.is_ready():
                self.video_recoder.start(self.file_path)

            frame = self.env.render(
                mode=self.mode, **self.render_kwargs)
            assert frame.dtype == np.uint8
            self.video_recoder.write_frame(frame)
        return result
    
    def render(self, mode='rgb_array', **kwargs):
        if self.video_recoder.is_ready():
            self.video_recoder.stop()
        return self.file_path
