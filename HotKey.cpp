#include <iostream>
#include "HotKey.h"


void HotKey::postKey(Gesture g) {
    switch (g) {
    case UP://��ʼ���������ƣ���ȡ״̬��Ȼ�������ƶ�������UP
        /*for (int i = 0;i < 3;i++) {
            mouse_event(MOUSEEVENTF_WHEEL, 0, 0, 120, 0);
        }*/
        mouse_event(MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_MOVE, 1500, 500, 0, 0);
        std::cout << "UP" << std::endl;
        break;
    case DOWN://���ĳ�ʼ���������ƶ�����ȡ�Ϸ������״̬��Ȼ�������ƶ�����������
        /*for (int i = 0;i < 3;i++) {
            mouse_event(MOUSEEVENTF_WHEEL, 0, 0, -120, 0);
        }*/
        mouse_event(MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_MOVE, 1500, 500, 0, 0);
        std::cout << "DOWN" << std::endl;
        break;
    case LEFT://���ĳ�ʼ��,�����ƶ�����ȡ״̬������ЧӦ�����ڵ�������ͷ������ʵ������,���ж�Moveʱ������
        keybd_event(VK_CONTROL, 0, 0, 0);
        keybd_event(VK_F4, 0, 0, 0);
        keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0);
        keybd_event(VK_F4, 0, KEYEVENTF_KEYUP, 0);
        mouse_event(MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_MOVE, 1500, 500, 0, 0);
        std::cout << "TO LEFT" << std::endl;
        break;
    case RIGHT:
        keybd_event(VK_CONTROL, 0, 0, 0);
        keybd_event(78, 0, 0, 0);
        keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0);
        keybd_event(78, 0, KEYEVENTF_KEYUP, 0);
        mouse_event(MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_MOVE, 1500, 500, 0, 0);
        std::cout << "TO RIGHT" << std::endl;
        break;
    case NOOPERATION:
        break;
    }
}
