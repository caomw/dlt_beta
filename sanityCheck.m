function bbox = sanityCheck(param, imSize)
    bbox = param(1:4, :)';
	w = imSize(2);
    h = imSize(1);
	
	bbox(:, 3) = min(bbox(:, 3), w);
	bbox(:, 4) = min(bbox(:, 4), h);
    
    bbox(:, 1) = max(bbox(:, 1) ,1);
    bbox(:, 1) = min(w-bbox(:, 3)+1, bbox(:, 1));
    bbox(:, 2) = max(bbox(:, 2), 1);
    bbox(:, 2) = min(h-bbox(:, 4)+1, bbox(:, 2));
	
	bbox = uint8(bbox);
end