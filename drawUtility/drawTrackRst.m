function [] = drawTrackRst(frame, bbox);
	% input - frame: normalized frame [0, 1]
	%       - param: affine matrix  
	clf
    imshow(uint8(frame));
	%bbox = param2bbox(param, size(frame), [227, 227]);   % get bbox
	rectangle('Position', [bbox(1:4)], 'LineWidth', 2.5, 'EdgeColor', 'r');
	drawnow;
end