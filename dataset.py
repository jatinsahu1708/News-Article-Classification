import cv2
import os

frame_size = (512,512)

def save_all_frames(video_path, dir_path, basename, ext='jpg'):
    cap = cv2.VideoCapture(video_path)
    print('start')
    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    n = 0

    while True:
        ret, frame = cap.read()
        if ret:
            if n % 100 == 0:
                #frame = cv2.resize(frame,frame_size)
                cv2.imwrite('{}_{}.{}'.format(base_path, str(n).zfill(digit), ext), frame)
                print('wrote')
            n += 1
        else:
            return

if __name__ == '__main__':
    save_all_frames('house_fire.mp4', 'frames', 'img')