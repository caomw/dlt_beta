function bbox = param2bbox(param, tmplsize)
    % bbox = [x0, y0, w, h, rot]
    bbox = zeros(5, size(param, 2));

    bbox(1,:) = round(param(1,:));
    bbox(2,:) = round(param(2,:));
    bbox(3,:) = param(3,:) * tmplsize(2);
    bbox(4,:) = param(5,:) * (tmplsize(1) / tmplsize(2)) .* bbox(3,:);
    bbox(5,:) = param(4,:);
    bbox(3,:) = round(bbox(3,:));
    bbox(4,:) = round(bbox(4,:));
    bbox = bbox';
end