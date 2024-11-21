import pygame
import time
from pymycobot import ElephantRobot
import threading

# elephant_client = ElephantRobot("172.30.21.69", 5001)
elephant_client = ElephantRobot("192.168.0.2", 5001)

try:
    elephant_client.start_client()
except Exception as e:
    print(f"Failed to start elephant client: {e}")
    exit(1)

flag='open'

# Pygameの初期化
pygame.init()

# ジョイスティックの初期化
pygame.joystick.init()

# 利用可能なジョイスティックの数を取得
joystick_count = pygame.joystick.get_count()

elephant_client.set_gripper_mode(1)

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

# ロボットが動作中かどうかを管理するフラグ
is_moving = False

# ゲームループ
running = True
all_state = elephant_client.get_angles()
print(f"Current coordinates: {all_state}")

joint1 = all_state[0]
joint2 = all_state[1]
joint3 = all_state[2]
joint4 = all_state[3]
joint5 = all_state[4]
joint6 = all_state[5]
while running:

    event= pygame.event.get()
    if elephant_client.check_running()==False:
        # 軸の状態を読み取る

        # 初期値
        if joystick.get_button(10) == 1:
            elephant_client.write_angles([0, -90, 0, -30, -90, 20], 1000)
        # グリッパー
        if joystick.get_button(9) == 1:
            if flag=='close':
                elephant_client.set_digital_out(16, 1) # Close the gripper
                time.sleep(1)
                elephant_client.set_digital_out(17, 0) # IO restores low level
                flag='open'
            else:
                elephant_client.set_digital_out(16, 0) #IO restores low level
                time.sleep(1)
                elephant_client.set_digital_out(17, 1) # Open the gripper
                flag='close'

            elephant_client.set_digital_out(17, 1) # IO restores low level
        if joystick.get_button(4) == 1:
            try:
                # print('speed', elephant_client.check_running())

                move1 = joystick.get_axis(0)
                move2 = (joystick.get_axis(1) + 3.0517578125e-05)
                move3 = (joystick.get_axis(4) + 3.0517578125e-05)
                
                # ロボットに新しい座標を送信
                if move1!=0 or move2!=0 or move3!=0:
                    joint1 = joint1 + move1
                    joint2 = joint2 + move2
                    joint3 = joint3 + move3
                    print(f"Moving to: joint1={joint1}, joint2={joint2}, joint3={joint3}")
                    elephant_client.write_angles([joint1, joint2, joint3, joint4, joint5, joint6], 1000)
                    time.sleep(1/125)

            except Exception as e:
                print(f"Error during movement: {e}")

        elif joystick.get_button(5) == 1:
            try:
                move4 = joystick.get_axis(0)
                move5 = (joystick.get_axis(1) + 3.0517578125e-05)
                move6 = (joystick.get_axis(4) + 3.0517578125e-05)
                
                # ロボットに新しい座標を送信
                if move4!=0 or move5!=0 or move6!=0:
                    joint4 = joint4 + move4
                    joint5 = joint5 + move5
                    joint6 = joint6 + move6
                    print(f"Moving to: joint4={joint4}, joint5={joint5}, joint6={joint6}")
                    elephant_client.write_angles([joint1, joint2, joint3, joint4, joint5, joint6], 1000)
                    time.sleep(1/125)

            except Exception as e:
                print(f"Error during movement: {e}")

pygame.quit()


# ボタン確認
# import pygame

# # Pygameの初期化
# pygame.init()
# pygame.joystick.init()

# # ジョイスティックの初期化
# if pygame.joystick.get_count() > 0:
#     joystick = pygame.joystick.Joystick(0)
#     joystick.init()
#     print(f"Joystick initialized: {joystick.get_name()}")
# else:
#     print("No joystick detected.")
#     exit()

# # メインループ
# running = True
# while running:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False
        
#         # ボタンが押された場合
#         if event.type == pygame.JOYBUTTONDOWN:
#             print(f"Button {event.button} pressed.")
        
#         # ボタンが離された場合
#         if event.type == pygame.JOYBUTTONUP:
#             print(f"Button {event.button} released.")
