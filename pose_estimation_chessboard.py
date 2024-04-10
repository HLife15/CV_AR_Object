import numpy as np
import cv2 as cv

video_file = 'C:\\Users\\USER\\Desktop\\chessboard.wmv'
# 지난 과제에서 구했던 카메라 캘리브레이션 결과값
K = np.array([[1.26064501e+03, 0.00000000e+00, 7.44017741e+02],
[0.00000000e+00, 1.70245148e+03, 4.85183754e+02],
[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist_coeff = np.array([ 0.12038199, -0.19289355, -0.02137884, 0.0073491, 0.1622757 ])
board_pattern = (10, 7)
board_cellsize = 0.025
board_criteria = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

video = cv.VideoCapture(video_file)
assert video.isOpened(), 'Cannot read the given input, ' + video_file

star_lower = board_cellsize * np.array([[4.2, 2,  0], [4.5, 2.1, 0], [4.8, 2,  0], [4.7, 2.3, 0], [5, 2.6, 0], [4.6, 2.7, 0], [4.5, 3, 0], [4.4, 2.7, 0], [4, 2.6, 0], [4.3, 2.3, 0]])
star_upper = board_cellsize * np.array([[4.2, 2,  -1.5], [4.5, 2.1, -1.5], [4.8, 2,  -1.5], [4.7, 2.3, -1.5], [5, 2.6, -1.5], [4.6, 2.7, -1.5], [4.5, 3, -1.5], [4.4, 2.7, -1.5], [4, 2.6, -1.5], [4.3, 2.3, -1.5]])

obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])

while True:
    valid, img = video.read()
    if not valid:
        break

    success, img_points = cv.findChessboardCorners(img, board_pattern, board_criteria)
    if success:
        ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

        line_lower, _ = cv.projectPoints(star_lower, rvec, tvec, K, dist_coeff)
        line_upper, _ = cv.projectPoints(star_upper, rvec, tvec, K, dist_coeff)
        cv.fillPoly(img, [np.int32(line_lower)], (160, 160, 160))
        cv.fillPoly(img, [np.int32(line_upper)], (0, 255, 255))

        R, _ = cv.Rodrigues(rvec)
        p = (-R.T @ tvec).flatten()
        info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
        cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0))

    cv.imshow('Star on Chessboard', img)
    key = cv.waitKey(10)
    if key == ord(' '):
        key = cv.waitKey()
    if key == 27: # ESC
        break

video.release()
cv.destroyAllWindows()
