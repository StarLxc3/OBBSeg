import os
import cv2
import numpy as np
from scipy.spatial.distance import cdist


def preprocess(path_src, dataset):
    print('process', path_src)
    path_dst = path_src.replace(dataset, dataset+'-Processed')

    path_src_folder = os.path.join(path_src,'Frame')
    path_dst_folder = os.path.join(path_dst,'Frame')
    for name in os.listdir(path_src_folder):
        coord_data = []  # 用于存储OBB坐标数据
        box_coords = []  # 用于存储box坐标数据
        path_src_frame_name = os.path.join(path_src,'Frame',name)
        path_dst_frame_name = os.path.join(path_dst,'Frame',name)
        image    = cv2.imread(path_src_frame_name)
        image    = cv2.resize(image, (352,352), interpolation=cv2.INTER_LINEAR)
        
        # 匹配关联掩码
        base_name = os.path.splitext(name)[0]
        src_gt_dir = os.path.join(path_src, 'GT')
        
        # 查找所有匹配的掩码文件
        if dataset in  ['Nuclei_Pre']:
            mask_files = [f for f in os.listdir(src_gt_dir) 
                if f.startswith(base_name)]
        elif dataset in  ['SUN-SEG_Pre']:
            mask_files = [f for f in os.listdir(src_gt_dir) 
                if f.startswith(base_name+'_mask.')]
        elif dataset in  ['ISIC2018_Pre','BUSI_Pre']:
            mask_files = [f for f in os.listdir(src_gt_dir) 
                if f.startswith(base_name+'_')]
        else:
            mask_files = [f for f in os.listdir(src_gt_dir) 
                        if f.startswith(base_name+'.')]
        # 合并处理多掩码
        combined_mask = np.zeros((352, 352), dtype=np.uint8)
        
        for idx, mask_name in enumerate(mask_files, 1):
            mask_path = os.path.join(src_gt_dir, mask_name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                print(f"Warning: Failed to read mask {mask_path}")
                continue

            resized_mask = cv2.resize(mask, (352, 352), interpolation=cv2.INTER_NEAREST)
            combined_mask = np.maximum(combined_mask, resized_mask)

        contours = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        obb      = np.zeros_like(combined_mask, dtype=np.uint8)
        box     = np.zeros_like(combined_mask, dtype=np.uint8)
        scribble_mask = np.zeros_like(combined_mask, dtype=np.uint8)
        circle_mask = np.zeros_like(combined_mask, dtype=np.uint8)
        point_mask = np.zeros_like(combined_mask, dtype=np.uint8)  # 添加中心点掩码

        for contour in contours:
            rect = cv2.minAreaRect(contour)
            (cx, cy), (width, height), angle = rect
            # 仅当长和宽都大于5时进行处理
            if width > 5 and height > 5:
                # AABB
                x, y, w, h = cv2.boundingRect(contour)
                x, y, w, h = int(x), int(y), int(w), int(h)
                x_min, y_min = max(0, x), max(0, y)
                x_max, y_max = min(351, x + w), min(351, y + h)
                box[y_min:y_max, x_min:x_max] = 255
                
                # OBB:
                obb_points = cv2.boxPoints(rect)
                # obb_points = np.int0(obb_points)
                obb_points = obb_points.astype(np.int32)  
                cv2.fillConvexPoly(obb, obb_points, color=255)
                obb_points = ';'.join([','.join(map(str, point)) for point in obb_points])
                coord_data.append(f"{obb_points}\n")
                
                # Scribble
                theta = np.deg2rad(angle)
                vx, vy = np.cos(theta), np.sin(theta)  
                
                points = contour.squeeze()
                if len(points) < 3:  
                    continue
                    
                proj = points.dot([vx, vy])
                start_idx = np.argmin(proj)
                end_idx = np.argmax(proj)
                P0 = points[start_idx]  
                P2 = points[end_idx]    
                M = np.array(contour.squeeze().mean(axis=0), dtype=np.float32)
                nx, ny = -vy, vx
                t_values = np.linspace(-0.3, 0.3, 20)  
                candidate_pts = np.array([M + t * np.array([nx, ny]) * min(width, height) for t in t_values])
                dists = cdist(candidate_pts, points)
                min_dists = np.min(dists, axis=1)
                P1 = candidate_pts[np.argmax(min_dists)] 
                
                scribble_points = []
                for t in np.linspace(0, 1, 20): 
                    x = (1-t)**2 * P0[0] + 2*(1-t)*t*P1[0] + t**2*P2[0]
                    y = (1-t)**2 * P0[1] + 2*(1-t)*t*P1[1] + t**2*P2[1]
                    scribble_points.append([int(x), int(y)])
                
                for i in range(len(scribble_points) - 1):
                    cv2.line(scribble_mask, 
                            tuple(scribble_points[i]), 
                            tuple(scribble_points[i+1]), 
                            255, 5) 
                    
                # Circle
                hull = cv2.convexHull(contour)
                if len(hull) >= 5:
                    ellipse = cv2.fitEllipse(hull)
                    (cx, cy), (major_axis, minor_axis), angle = ellipse
                    major_axis = max(5, major_axis) 
                    minor_axis = max(5, minor_axis)
                    cv2.ellipse(circle_mask, 
                               (int(cx), int(cy)), 
                               (int(major_axis/2*1.2), int(minor_axis/2*1.2)),
                               angle, 0, 360, 255, -1)
                else:
                    rect = cv2.minAreaRect(hull)
                    circle_box = cv2.boxPoints(rect)
                    circle_box = circle_box.astype(np.int32)  
                    cv2.fillConvexPoly(circle_mask, circle_box, 255)
            
            # Point
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                contour_points = contour.squeeze(1)
                if len(contour_points) > 0:
                    cX = int(contour_points[:, 0].mean())
                    cY = int(contour_points[:, 1].mean())
                else:
                    continue

            cX = max(0, min(cX, combined_mask.shape[1] - 1))
            cY = max(0, min(cY, combined_mask.shape[0] - 1))
            
            # 在中心点掩码中设置点 (2像素半径确保可见)
            cv2.circle(point_mask, (cX, cY), 5, 255, -1)
                
        
        os.makedirs(path_dst_folder, exist_ok=True)
        cv2.imwrite(path_dst_frame_name, image)
        
        os.makedirs(path_dst_folder.replace('Frame','GT'), exist_ok=True)
        cv2.imwrite(path_dst_frame_name.replace('Frame', 'GT').replace('jpg','png'), combined_mask)
        
        os.makedirs(path_dst_folder.replace('Frame','OBB'), exist_ok=True)
        cv2.imwrite(path_dst_frame_name.replace('Frame','OBB').replace('jpg','png'), obb)

        # save AABB
        os.makedirs(os.path.join(path_dst, 'Box'), exist_ok=True)
        cv2.imwrite(path_dst_frame_name.replace('Frame', 'Box').replace('jpg','png'), box)

        # save obb
        os.makedirs(os.path.join(path_dst, 'Coord'), exist_ok=True)
        coord_file = os.path.join(path_dst, 'Coord', name.replace('.jpg', '.txt').replace('.png', '.txt'))
        with open(coord_file, "w") as f:
            f.writelines(coord_data)
            
        # save scribble
        os.makedirs(path_dst_folder.replace('Frame','Scribble'), exist_ok=True)
        cv2.imwrite(path_dst_frame_name.replace('Frame', 'Scribble').replace('jpg','png'), scribble_mask)
        
        # save circle
        os.makedirs(path_dst_folder.replace('Frame','Circle'), exist_ok=True)
        cv2.imwrite(path_dst_frame_name.replace('Frame', 'Circle').replace('jpg','png'), circle_mask)

        # save point
        os.makedirs(path_dst_folder.replace('Frame','Point'), exist_ok=True)
        cv2.imwrite(path_dst_frame_name.replace('Frame', 'Point').replace('jpg','png'), point_mask)