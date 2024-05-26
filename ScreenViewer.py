import numpy as np
import win32gui
import win32ui, win32con
from threading import Thread, Lock
import time
from ctypes import windll



'''借鉴PoE AI Part 4: Real-time Screen Capture and Plumbing'''
'''存在截图黑屏问题,解决借鉴http://t.csdnimg.cn/lQZgt PyWin32无法捕获某些窗口，导致黑屏(Python，Windows)'''
# 定义一个屏幕捕获类
class ScreenViewer:

    def __init__(self):
        self.mut = Lock()  # 创建一个互斥锁，用于同步访问共享资源
        self.hwnd = None   # hwnd 是窗口句柄
        self.its = None     # 上一次图像的时间戳
        self.i0 = None      # i0 是最新的图像
        self.i1 = None      # i1 用作临时变量
        self.cl = False     # 继续循环的标志
        # 窗口左上角和右下角的坐标
        self.l, self.t, self.r, self.b = 0, 0, 0, 0
        # 窗口左侧和顶部的边框宽度
        self.bl, self.bt, self.br, self.bb = 0, 0, 0, 0

    # 获取指定窗口的句柄
    # wname: 窗口标题
    # 返回： 成功找到窗口返回 True，否则返回 False
    def GetHWND(self, wname):
        self.hwnd = win32gui.FindWindow(None, wname)  # 查找窗口
        win32gui.SetForegroundWindow(self.hwnd)
        if self.hwnd == 0:  # 如果找不到窗口
            self.hwnd = None  # 设置 hwnd 为 None
            return False  # 返回 False
        self.l, self.t, self.r, self.b = win32gui.GetWindowRect(self.hwnd)  # 获取窗口矩形坐标
        return True  # 返回 True

    # 获取窗口的最新图像
    def GetScreen(self):
        while self.i0 is None:  # 直到图像被捕获
            pass
        self.mut.acquire()  # 获取互斥锁
        s = self.i0  # 获取图像
        self.mut.release()  # 释放互斥锁
        return s

    # 获取窗口的最新图像和时间戳
    def GetScreenWithTime(self):
        while self.i0 is None:  # 直到图像被捕获
            pass
        self.mut.acquire()  # 获取互斥锁
        s = self.i0  # 获取图像
        t = self.its  # 获取时间戳
        self.mut.release()  # 释放互斥锁
        return s, t  # 返回图像和时间戳

    # 获取窗口图像
    # 获取由 self.hwnd 引用的窗口的屏幕
    def GetScreenImg(self):
        windll.user32.SetProcessDPIAware()
        if self.hwnd is None:
            raise Exception("HWND is none. HWND not called or invalid window name provided.")
        # 获取窗口的矩形坐标
        self.l, self.t, self.r, self.b = win32gui.GetWindowRect(self.hwnd)
        # print(self.l, self.t, self.r, self.b)
        # 移除窗口周围的边框（每边8像素）
        # 从左边和右边减去24像素（16像素边框+8像素内边距）
        w = self.r - self.l - self.br - self.bl
        # 移除顶部和底部的边框（顶部31像素，底部8像素）
        # 从底部减去51像素（39像素底部边框+12像素内边距）
        h = self.b - self.t - self.bt - self.bb
   
        # 获取窗口的设备上下文（DC）
        hwnd_dc = win32gui.GetWindowDC(self.hwnd)
        # 从设备上下文创建一个兼容的DC
        mfc_dc = win32ui.CreateDCFromHandle( hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()
        # 创建一个兼容的位图
        dataBitMap = win32ui.CreateBitmap()
        # 创建一个与目标DC兼容的位图
        dataBitMap.CreateCompatibleBitmap(mfc_dc, w, h)
        # 将位图选入cDC中
        save_dc.SelectObject(dataBitMap)
        result = windll.user32.PrintWindow(self.hwnd, save_dc.GetSafeHdc(), 3)
        bmpinfo = dataBitMap.GetInfo()
        bmpstr =dataBitMap.GetBitmapBits(True)
        img = np.frombuffer(bmpstr, dtype=np.uint8).reshape((bmpinfo["bmHeight"], bmpinfo["bmWidth"], 4))
        img = np.ascontiguousarray(img)[..., :-1]  # make image C_CONTIGUOUS and drop alpha channel
        if not result:  # result should be 1
            win32gui.DeleteObject(dataBitMap.GetHandle())
            save_dc.DeleteDC()
            mfc_dc.DeleteDC()
            win32gui.ReleaseDC(self.hwnd, hwnd_dc)
            raise RuntimeError(f"Unable to acquire screenshot! Result: {result}")
        return  img

    def Start(self):
        # 如果 self.hwnd 是 None：
        # 返回 False
        #if self.hwnd is None:
        #    return False
        self.cl = True  # 设置继续循环标志为 True
        thrd = Thread(target = self.ScreenUpdateT)  # 创建一个线程，目标函数是 ScreenUpdateT
        thrd.start()  # 启动线程
        return True  # 返回 True

    # 停止捕获屏幕图像的异步线程
    def Stop(self):
        self.cl = False  # 设置继续循环标志为 False

    # 用于捕获屏幕图像的线程
    def ScreenUpdateT(self):
        # 持续更新屏幕直到被终止
        while self.cl:
            self.i1 = self.GetScreenImg()  # 获取屏幕图像
            self.mut.acquire()  # 获取互斥锁
            self.i0 = self.i1  # 以线程安全的方式更新最新的图像
            # print(self.i0.shape)
            self.its = time.time()  # 更新时间戳
            self.mut.release()  # 释放互斥锁

import cv2
if __name__ == "__main__":
    
    sv = ScreenViewer()
    sv.GetHWND('雷电模拟器')
    sv.Start()
    time.sleep(0.1)
    sv.Stop()
    blood_line=sv.i0[100:-90,181:-221,:]
    print(blood_line.shape)
    # cv2.imshow('img',blood_line)
   
    # cv2.waitKey(0)
    