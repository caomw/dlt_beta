function [] = drawTrackRst(frame, param);
	% input - frame: normalized frame [0, 1]
	%       - param: affine matrix  
	clf
    imshow(uint8(frame * 255)); % restore value to [0, 255]
	bbox = param2bbox(param, size(frame), [227, 227]);   % get bbox
	rectangle('Position', [bbox(1:4)], 'LineWidth', 2.5, 'EdgeColor', 'r');
	drawnow;
end