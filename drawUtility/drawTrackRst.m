function [] = drawTrackRst(frame, param);
	% input - frame: normalized frame [0, 1]
	%       - param: affine matrix  
	imshow(uint8(frame * 255)); % restore value to [0, 255]
	hold
	bbox = param2bbox(param, [32, 32]);   % get bbox
	rectangle('Position', [bbox(1:4)], 'LineWidth', 2.5, 'EdgeColor', 'r');
	hold off
end