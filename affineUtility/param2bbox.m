function bbox = param2bbox(param, imSize, tmplsize)
    % bbox = [x0, y0, w, h, rot]
    % param = affine geometric parameter: [dx, dy, scale, theta, aspect, skew]
    bbox = zeros(5, size(param, 2));
    w = imSize(2);
    h = imSize(1);

    bbox(1,:) = round(param(1,:));
    bbox(2,:) = round(param(2,:));
    bbox(3,:) = param(3,:) * tmplsize(2);
    bbox(4,:) = param(5,:) * (tmplsize(1) / tmplsize(2)) .* bbox(3,:);
    bbox(5,:) = param(4,:);
    bbox(3,:) = min(w, round(bbox(3,:)));
    bbox(4,:) = min(h, round(bbox(4,:)));
    bbox(1,:) = bbox(1,:)-bbox(3,:)/2;
    bbox(2,:) = bbox(2,:)-bbox(4,:)/2;
    bbox = uint8(bbox');
    
    bbox(:, 1) = max(bbox(:, 1) ,1);
    bbox(:, 1) = min(w-bbox(:, 3)+1, bbox(:, 1));
    bbox(:, 2) = max(bbox(:, 2), 1);
    bbox(:, 2) = min(h-bbox(:, 4)+1, bbox(:, 2));
end

%{
function bbox = sanityCheck(bbox, imSize)
    w = imSize(1);
    h = imSize(2);
    
    bbox(:, 1) = max(bbox(:, 1) ,1);
    bbox(:, 1) = min(w-bbox(:, 3)+1, bbox(:, 1) + bbox(:, 3));
    bbox(:, 2) = max(bbox(:, 2), 1);
    bbox(:, 2) = min(h-bbox(:, 4)+1, bbox(:, 2) + bbox(:, 4));
end
%}