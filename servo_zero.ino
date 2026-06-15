/*
 * MeArm V1.0 — 舵机调零程序
 * ==========================
 * 用于机械组装时确认舵机物理零位。
 * 上电后所有舵机归位到 HOME=90°（夹爪 25°），保持不动。
 *
 * 接线 (同主程序):
 *   D11 — 底座舵机 (Base / Middle)
 *   D10 — 左臂舵机 (Left)
 *   D9  — 右臂舵机 (Right)
 *   D6  — 夹爪舵机 (Claw)
 *
 * 使用方法:
 *   1. 上传此程序到 Arduino Uno
 *   2. 舵机自动归 90°（夹爪 25°）
 *   3. 断电，安装舵盘使机械臂处于直立零位
 *   4. 确认完成后上传主程序 main.ino
 *
 * 编译: Arduino IDE 打开此文件, 选择 Uno 板, 编译上传
 */

#include <Servo.h>

Servo middle, left, right, claw;  // 4 个舵机对象

void setup()
{
  Serial.begin(9600);
  middle.attach(11);  // 底座舵机
  left.attach(10);    // 左臂舵机
  right.attach(9);    // 右臂舵机
  claw.attach(6);     // 夹爪舵机

  middle.write(90);   // 底座归中
  left.write(90);     // 左臂归中
  right.write(90);    // 右臂归中
  claw.write(25);     // 夹爪半开
  delay(300);         // 等待舵机到位
}

void loop()
{
  // 保持当前位置不动
  middle.write(90);
  left.write(90);
  right.write(90);
  claw.write(25);
  delay(300);
}
