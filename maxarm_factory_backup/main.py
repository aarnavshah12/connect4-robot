import timeimport uctypes, gcimport Hiwonder_wifi_ble as HW_wbfrom Key import Keyfrom Led import LEDfrom USBDevice import *from Buzzer import Buzzerfrom espmax import ESPMaxfrom PWMServo import PWMServofrom micropython import constfrom machine import Pin, ADC, Timerfrom RobotControl import RobotControlfrom SuctionNozzle import SuctionNozzlefrom BusServo import BusServo, have_got_servo_posprint("Please wait...")key = Key()led = LED()buzzer = Buzzer()pwm = PWMServo()pwm.work_with_time()bus_servo = BusServo()arm = ESPMax(bus_servo)nozzle = SuctionNozzle()robot = RobotControl(nozzle)ble = HW_wb.Hiwonder_wifi_ble(HW_wb.MODE_BLE_SLAVE, name = 'MaxArm')ble.set_led_key_io(led=26,key=25)arm.go_home()print_en = Truenozzle_st = Falsenozzle_angle = 0(x,y,z) = arm.ORIGINmove_sleep = time.ticks_ms()nozzle.set_angle(nozzle_angle)# 鼠标控制处理函数 def MouseHandle():  global x,y,z,nozzle_st  global nozzle_angle,move_sleep    BUTTON_L = 0x01  BUTTON_R = 0x02  BUTTON_M = 0x04    msg = USBDevice.get_mouse_msg()    if time.ticks_ms() >= move_sleep:     if msg == False:      return     mouse_msg = uctypes.struct(uctypes.addressof(bytes(msg))              ,{"button": uctypes.UINT8 | 5,'move_X': uctypes.INT8 | 6              , 'move_Y': uctypes.INT8 | 7,'wheel': uctypes.INT8 | 8})    if mouse_msg.button & BUTTON_M != 0: # 机械臂复位      nozzle.off()      arm.go_home(1500)      nozzle_angle = 0      nozzle_st = False      (x,y,z) = arm.ORIGIN      buzzer.setBuzzer(80)      nozzle.set_angle(nozzle_angle)      move_sleep = time.ticks_ms() + 1500          elif mouse_msg.wheel != 0: # Z轴控制      dz = -mouse_msg.wheel*2      if arm.set_position((x, y, z+dz), 20):        z += dz      move_sleep = time.ticks_ms() + 30        else:      if abs(mouse_msg.move_X) > abs(mouse_msg.move_Y):        if mouse_msg.button & BUTTON_L != 0: # 吸盘角度控制          nozzle_angle += int(mouse_msg.move_X/3)          if nozzle_angle > 90:nozzle_angle = 90          if nozzle_angle < -90:nozzle_angle = -90          nozzle.set_angle(nozzle_angle)          move_sleep = time.ticks_ms() + 30                  else:          dx = -int(mouse_msg.move_X/8) # X轴控制          if arm.set_position((x+dx, y, z), 20):            x += dx          move_sleep = time.ticks_ms() + 30              elif abs(mouse_msg.move_X) < abs(mouse_msg.move_Y):        if mouse_msg.button & BUTTON_R != 0: # 气泵控制          if time.ticks_ms() >= move_sleep:            nozzle_st = bool(1 - nozzle_st)            if nozzle_st:nozzle.on()            else:nozzle.off()            move_sleep = time.ticks_ms() + 300                  else:          if mouse_msg.button & BUTTON_L == 0:  # Y轴控制            dy = int(-mouse_msg.move_Y/8)            if arm.set_position((x, y+dy, z), 20):              y += dy            move_sleep = time.ticks_ms() + 30ble_buf = bytearray()ble_data = bytearray()
# 蓝牙回调函数
def ble_callback():
  global ble_buf
  global ble_data
  
  ble_buf += ble.ble_read() # 读取蓝牙发来的数据
  ble_data = ble_buf
  ble_buf = bytearray()

ble.ble_rx_irq(ble_callback)  

# 蓝牙控制处理函数def BleHandle():  global ble_data,reset_sleep,print_en  global x,y,z, nozzle_angle, move_sleep    de = 3  if time.ticks_ms() >= move_sleep:    if ble.mode == HW_wb.MODE_BLE_SLAVE:      ble_data_len = len(ble_data)      if ble_data_len >= 4:        if ble_data[0] == 0x55 and ble_data[1] == 0x55:          cmd = ble_data[3]          if cmd == 0x23: # 机械臂XYZ轴控制            if ble_data[4] < 0x80: dx = ble_data[4]             else: dx = (ble_data[4] - 256)            if ble_data[5] < 0x80: dy = ble_data[5]            else: dy = (ble_data[5] - 256)                       if ble_data[6] < 0x80: dz = ble_data[6]            else: dz = (ble_data[6] - 256)                        if dx== -1 and dy== -1 and dz== -1: # 机械臂复位              arm.go_home(1500)              nozzle.off()              buzzer.setBuzzer(80)              nozzle_angle = 0              nozzle.set_angle(0)              (x,y,z) = arm.ORIGIN              move_sleep = time.ticks_ms() + 1500                          else: # 机械臂移动              if arm.set_position((x+dx*de,y+dy*de,z+dz*de), 20): # 运动学求解，有解则移动机械臂                x += dx*de                 y += dy*de
                z += dz*de
                print_en = True
              else:  # 无解则蜂鸣器提示
                if print_en:
                  buzzer.setBuzzer(20)
                  print_en = False
                              move_sleep = time.ticks_ms() + 30                      if cmd == 0x24:# 吸盘角度控制            if ble_data[4] < 0x80: da = ble_data[4]             else: da = (ble_data[4] - 256)            nozzle_angle += da*3            nozzle_angle = 90 if nozzle_angle > 90 else nozzle_angle            nozzle_angle = -90 if nozzle_angle < -90 else nozzle_angle            nozzle.set_angle(nozzle_angle, 80)            move_sleep = time.ticks_ms() + 30                      elif cmd == 0x25:# 吸盘控制            st = ble_data[4]            if st == 1: nozzle.on()            elif st == 0: nozzle.off()
            move_sleep = time.ticks_ms() + 300
            
          elif cmd == 0x06:#动作组运行
            name = ble_data[4]
            times = ble_data[5]
            if times == 0:times = 100000
            robot.runActionGroup(str(name), times)                      elif cmd == 0x07:#动作组停止            robot.stopActionGroup()
    
  ble_data = bytearray()


def main(t):
  global x,y,z, move_sleep, nozzle_angle
  #在定时器中断中完成的，不要出现死循环和过大的延时函数
  
  gc.collect()
  key.run_loop()
  USBDevice.run_loop()    BleHandle()  MouseHandle()  if key.down_up(): # 短按key1执行100号动作组    if time.ticks_ms() >= move_sleep:      robot.runActionGroup("100")      move_sleep = time.ticks_ms() + 1500      if key.down_long(): # 长按key1停止运行动作组    if time.ticks_ms() >= move_sleep:      robot.stopActionGroup()      move_sleep = time.ticks_ms() + 300print("Start")tim = Timer(2)tim.init(period=15, mode=Timer.PERIODIC, callback=main)