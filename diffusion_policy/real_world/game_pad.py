import multiprocessing as mp
import numpy as np
import time
import pygame
import time
# from spnav import spnav_open, spnav_poll_event, spnav_close, SpnavMotionEvent, SpnavButtonEvent
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer

class Control(mp.Process):
    def __init__(self, 
            shm_manager, 
            get_max_k=30, 
            frequency=200,
            max_value=500, 
            deadzone=(0,0,0,0,0,0), 
            dtype=np.float32,
            n_buttons=2,
            ):
        """
        Continuously listen to 3D connection space naviagtor events
        and update the latest state.

        max_value: {300, 500} 300 for wired version and 500 for wireless
        deadzone: [0,1], number or tuple, axis with value lower than this value will stay at 0
        
        front
        z
        ^   _
        |  (O) space mouse
        |
        *----->x right
        y
        """
        super().__init__()
        if np.issubdtype(type(deadzone), np.number):
            deadzone = np.full(6, fill_value=deadzone, dtype=dtype)
        else:
            deadzone = np.array(deadzone, dtype=dtype)
        assert (deadzone >= 0).all()

        # copied variables
        self.frequency = frequency
        self.max_value = max_value
        self.dtype = dtype
        self.deadzone = deadzone
        self.n_buttons = n_buttons
        # self.motion_event = SpnavMotionEvent([0,0,0], [0,0,0], 0)
        # self.button_state = defaultdict(lambda: False)
        self.tx_zup_spnav = np.array([
            [0,0,-1],
            [1,0,0],
            [0,1,0]
        ], dtype=dtype)

        example = {
            # 3 translation, 3 rotation, 1 period
            'motion_event': np.zeros((7,), dtype=np.int64),
            # left and right button
            'button_state': np.zeros((n_buttons,), dtype=bool),
            'receive_timestamp': time.time()
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager, 
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        # shared variables
        self.ready_event = mp.Event()
        self.stop_event = mp.Event()
        self.ring_buffer = ring_buffer

    # ======= get state APIs ==========

    def get_motion_state(self):
        # TODO:たぶんここの形確認する必要ある
        state = self.ring_buffer.get()
        print('ringbuf', state)
        state = np.array(state['motion_event'][:6], 
            dtype=self.dtype) / self.max_value
        is_dead = (-self.deadzone < state) & (state < self.deadzone)
        state[is_dead] = 0
        return state
    
    def get_motion_state_transformed(self):
        """
        Return in right-handed coordinate
        z
        *------>y right
        |   _
        |  (O) space mouse
        v
        x
        back

        """
        state = self.get_motion_state()
        # print('state', state)
        tf_state = np.zeros_like(state)
        tf_state[:3] = state[:3]
        tf_state[3:] = state[3:]
        # tf_state[:3] = self.tx_zup_spnav @ state[:3]
        # tf_state[3:] = self.tx_zup_spnav @ state[3:]
        return tf_state

    # TODO:たぶんいらない
    # def get_button_state(self):
    #     state = self.ring_buffer.get()
    #     return state['button_state']
    
    # def is_button_pressed(self, button_id):
    #     return self.get_button_state()[button_id]
    
    #========== start stop API ===========

    def start(self, wait=True):
        super().start()
        if wait:
            self.ready_event.wait()
    
    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.join()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= main loop ==========
    def run(self):
        # Pygameの初期化
        pygame.init()

        # ジョイスティックの初期化
        pygame.joystick.init()

        # 利用可能なジョイスティックの数を取得
        joystick_count = pygame.joystick.get_count()

        if joystick_count == 0:
            print("コントローラーが接続されていません。")
            exit(1)
        else:
            # 最初のジョイスティックを使用
            joystick = pygame.joystick.Joystick(0)
            joystick.init()
            
            print(f"コントローラー名: {joystick.get_name()}")
            print(f"軸の数: {joystick.get_numaxes()}")
            print(f"ボタンの数: {joystick.get_numbuttons()}")
            print(f"ハットスイッチの数: {joystick.get_numhats()}")


        try:
            motion_event = np.zeros((7,), dtype=np.float64)
            button_state = np.zeros((self.n_buttons,), dtype=bool)
            # send one message immediately so client can start reading
            self.ring_buffer.put({
                'motion_event': motion_event,
                'button_state': button_state,
                'receive_timestamp': time.time()
            })
            self.ready_event.set()

            while not self.stop_event.is_set():
                event = pygame.event.get()
                receive_timestamp = time.time()
                if joystick.get_button(4) == 1:
                    motion_event[0] = joystick.get_axis(0)
                    # motion_event[1] = joystick.get_axis(1)#(joystick.get_axis(1) + 3.0517578125e-05)
                    motion_event[1] = (joystick.get_axis(1) + 3.0517578125e-05)
                    # motion_event[2] = joystick.get_axis(4)#(joystick.get_axis(4) + 3.0517578125e-05)
                    motion_event[2] = (joystick.get_axis(4) + 3.0517578125e-05)
                    # print('input', motion_event)
                    self.ring_buffer.put({
                        'motion_event': motion_event,
                        'button_state': button_state,
                        'receive_timestamp': receive_timestamp
                    })
                    time.sleep(0.1)
                    
                elif joystick.get_button(5) == 1:
                    # motion_event[:3] = event.translation
                    motion_event[3] = joystick.get_axis(0)
                    # motion_event[4] = joystick.get_axis(1)#(joystick.get_axis(1) + 3.0517578125e-05)
                    # motion_event[5] = joystick.get_axis(4)#(joystick.get_axis(4) + 3.0517578125e-05)
                    motion_event[4] = (joystick.get_axis(1) + 3.0517578125e-05)
                    motion_event[5] = (joystick.get_axis(4) + 3.0517578125e-05)
                    print('input', motion_event)
                    self.ring_buffer.put({
                        'motion_event': motion_event,
                        'button_state': button_state,
                        'receive_timestamp': receive_timestamp
                    })
                    time.sleep(0.1)
                # else:
                #     motion_event = np.zeros((7,), dtype=np.float64)
                #     self.ring_buffer.put({
                #         'motion_event': motion_event,
                #         'button_state': button_state,
                #         'receive_timestamp': receive_timestamp
                #     })
                    # motion_event[6] = event.period
                # elif isinstance(event, SpnavButtonEvent):
                #     button_state[event.bnum] = event.press
                # else:
                #     # finish integrating this round of events
                #     # before sending over
                #     print('input', motion_event)
                #     self.ring_buffer.put({
                #         'motion_event': motion_event,
                #         'button_state': button_state,
                #         'receive_timestamp': receive_timestamp
                #     })
                #     # print('aadejgjoewjgojwoep')
                #     # time.sleep(1/self.frequency)
                #     time.sleep(0.1)
        finally:
            pygame.quit()
