function plotrect(classifiers, II, actual_images, face_rects, non_face_rects, DOFULL)
    % Divide images in quadrants
    im1 = II(1:240,1:360,:);
    im2 = II(1:240,361:end,:);
    im3 = II(241:end,1:360,:);
    im4 = II(241:end,361:end,:);
    
    II_stack = cat(3,im1,im2,im3,im4);
    K = size(II_stack, 3);
    
    % Divide images in quadrants
    im1 = actual_images(1:240,1:360,:);
    im2 = actual_images(1:240,361:end,:);
    im3 = actual_images(241:end,1:360,:);
    im4 = actual_images(241:end,361:end,:);

    actualim_stack = cat(3,im1,im2,im3,im4);

    % Gather rectangles
    f1 = face_rects(:,1:4);
    f2 = face_rects(:,5:8);
    f3 = face_rects(:,9:12);
    f4 = face_rects(:,13:16);
    
    nf1 = non_face_rects(:,1:4);
    nf2 = non_face_rects(:,5:8);
    nf3 = non_face_rects(:,9:12);
    nf4 = non_face_rects(:,13:16);
    
    actual_rects = cat(1, f1,f2,f3,f4,nf1,nf2,nf3,nf4);    
    rects_row = reshape(actual_rects', 1, []);
    rect_mat = repmat(rects_row, K, 1);
    NR = size(rect_mat, 2)/4;

    F = zeros(K, NR);
    
    %Find best rectangles
    for t = 1:40^DOFULL
        disp(t);
        xmin = classifiers(t,1);
        ymin = classifiers(t,2);
        wid = classifiers(t,3);
        hgt = classifiers(t,4);
        vert = classifiers(t,5);
        order = classifiers(t,6);
        theta = classifiers(t,7);
        pola = classifiers(t,8);

        FR = [xmin,ymin,wid,hgt];
        F_curr = rectfeature(II_stack,rect_mat,FR,order,vert);
        F = F + F_curr * pola;
    end
    
    % Move rects into first quad
    vert_shift = repmat(240,168,1);
    horz_shift = repmat(360,168,1);
    actual_shifted_rects = actual_rects;
    actual_shifted_rects(169:168*2,1) = actual_rects(169:168*2,1) - horz_shift;
    actual_shifted_rects(169+168:168*3,2) = actual_rects(169+168:168*3,2) - vert_shift;
    actual_shifted_rects(169+168*2:168*4,2) = actual_rects(169+168*2:168*4,2) - vert_shift;
    actual_shifted_rects(169+168*2:168*4,1) = actual_rects(169+168*2:168*4,1) - horz_shift;

    pred_rects = zeros(K,4);
    for i = 1:K
        best_val = min(F(i,:));
        best_idx = find(best_val == F(i,:));
        pred_rects(i,:) = rects_row((best_idx-1)*4+1:(best_idx-1)*4 + 4);

        figure(1); hold off;
        imagesc(actualim_stack(:,:,i)); hold on;
        plotrect(pred_rects(i,:), 'r');
        plotrect(actual_shifted_rects(i,:), 'g');
        
        g = input(sprintf('Hit return for image: %d quadrant: %d', mod(i-1,K/4)+1, (floor(i/(K/4)) + 1)));
    end   

end