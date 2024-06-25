from typing import Optional, Callable, Generator
import numpy as np
import av
from diffusion_policy.common.timestamp_accumulator import get_accumulate_timestamp_idxs

def read_video(
        video_path: str, dt: float,
        video_start_time: float=0.0, 
        start_time: float=0.0,
        img_transform: Optional[Callable[[np.ndarray], np.ndarray]]=None,
        thread_type: str="AUTO",
        thread_count: int=0,
        max_pad_frames: int=10
        ) -> Generator[np.ndarray, None, None]:
    frame = None
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        stream.thread_type = thread_type
        stream.thread_count = thread_count
        next_global_idx = 0
        for frame_idx, frame in enumerate(container.decode(stream)):
            # The presentation time in seconds for this frame.
            since_start = frame.time
            frame_time = video_start_time + since_start
            local_idxs, global_idxs, next_global_idx \
                = get_accumulate_timestamp_idxs(
                # only one timestamp
                timestamps=[frame_time],
                start_time=start_time,
                dt=dt,
                next_global_idx=next_global_idx
            )
            if len(global_idxs) > 0:
                array = frame.to_ndarray(format='rgb24')
                img = array
                if img_transform is not None:
                    img = img_transform(array)
                for global_idx in global_idxs:
                    yield img
    # repeat last frame max_pad_frames times
    array = frame.to_ndarray(format='rgb24')
    img = array
    if img_transform is not None:
        img = img_transform(array)
    for i in range(max_pad_frames):
        yield img

class VideoRecorder:
    def __init__(self,
        fps,
        codec,
        input_pix_fmt,
        # options for codec
        **kwargs
    ):
        """
        input_pix_fmt: rgb24, bgr24 see https://github.com/PyAV-Org/PyAV/blob/bc4eedd5fc474e0f25b22102b2771fe5a42bb1c7/av/video/frame.pyx#L352
        """

        self.fps = fps
        self.codec = codec
        self.input_pix_fmt = input_pix_fmt
        self.kwargs = kwargs
        # runtime set
        self._reset_state()
    
    def _reset_state(self):
        self.container = None
        self.stream = None
        self.shape = None
        self.dtype = None
        self.start_time = None
        self.next_global_idx = 0
    
    @classmethod
    def create_h264(cls,
            fps,
            codec='h264',
            input_pix_fmt='rgb24',
            output_pix_fmt='yuv420p',
            crf=18,
            profile='high',
            **kwargs
        ):
        obj = cls(
            fps=fps,
            codec=codec,
            input_pix_fmt=input_pix_fmt,
            pix_fmt=output_pix_fmt,
            options={
                'crf': str(crf),
                'profile': profile
            },
            **kwargs
        )
        return obj


    def __del__(self):
        self.stop()

    def is_ready(self):
        return self.stream is not None

    def start(self, file_path, start_time=None):
        if self.is_ready():
            # if still recording, stop first and start anew.
            self.stop()

        self.container = av.open(file_path, mode='w')
        self.stream = self.container.add_stream(self.codec, rate=self.fps)
        codec_context = self.stream.codec_context
        for k, v in self.kwargs.items():
            setattr(codec_context, k, v)
        self.start_time = start_time
    
    def write_frame(self, img: np.ndarray, frame_time=None):
        if not self.is_ready():
            raise RuntimeError('Must run start() before writing!')
        
        n_repeats = 1
        if self.start_time is not None:
            local_idxs, global_idxs, self.next_global_idx \
                = get_accumulate_timestamp_idxs(
                # only one timestamp
                timestamps=[frame_time],
                start_time=self.start_time,
                dt=1/self.fps,
                next_global_idx=self.next_global_idx
            )
            # number of appearance means repeats
            n_repeats = len(local_idxs)
        
        if self.shape is None:
            self.shape = img.shape
            self.dtype = img.dtype
            h,w,c = img.shape
            self.stream.width = w
            self.stream.height = h
        assert img.shape == self.shape
        assert img.dtype == self.dtype

        frame = av.VideoFrame.from_ndarray(
            img, format=self.input_pix_fmt)
        for i in range(n_repeats):
            for packet in self.stream.encode(frame):
                self.container.mux(packet)

    def stop(self):
        if not self.is_ready():
            return

        # Flush stream
        for packet in self.stream.encode():
            self.container.mux(packet)

        # Close the file
        self.container.close()

        # reset runtime parameters
        self._reset_state()
