%function param = estwarp_condens_DLT(frame, param, opt)
	global caffe_batch_size;
	global object_class;
	global DEBUG;

	n = opt.numsample;
	sz = size(frame(:,:,1));
	if ~exist('selected_idx', 'var')
		selected_idx = [];
	end
	
	param.param = repmat(param.est(:), [1,n]);
	bbox = param.param(1:4, :) + randn(4,n).*repmat(opt.affsig(:),[1,n]);% + 0.5*repmat([opt.motion, 0, 0]',[1,n]);
	bbox = sanityCheck(bbox, sz);


	d = load('./caffe/ilsvrc_2012_mean');
	IMAGE_MEAN = d.image_mean;
	IMAGE_MEAN = imresize(IMAGE_MEAN, opt.tmplsize);

	% extract patch feeded into caffe
	% images: 227 x 227 x 3 x n, type should be single
	% bbox: n-by-4
	images = zeros(opt.tmplsize(1), opt.tmplsize(2), 3, n, 'single');
	X = bbox(:,1)+bbox(:,3)/2;
	Y = bbox(:,2)+bbox(:,4)/2;
	
	tic;
	for i = 1:n
		images(:,:,:,i) = imresize(frame(bbox(i,2):bbox(i,2)+bbox(i,4)-1, bbox(i,1):bbox(i,1)+bbox(i,3)-1, :), opt.tmplsize);
	end
	images = bsxfun( @minus, images(:,:,[3 2 1],:), IMAGE_MEAN );
	images = permute(images, [2 1 3 4]);
	toc;

	epoch = ceil(n / caffe_batch_size);
	confidence = zeros(21, n);

	tic;
	for e = 1:epoch
		start = (e-1)*caffe_batch_size + 1;
		if e == epoch
			last = n;
		else
			last = start + caffe_batch_size - 1;
		end
		input_data = {images(:,:,:,start:last)};
		scores = caffe('forward', input_data);
		confidence(:, start:last) = reshape(scores{1}, [21, caffe_batch_size]);
	end
	toc;

	confidence = confidence(object_class+1,:)';
	disp(max(confidence));
	selected_idx = find(confidence > 0.7);
	if(length(selected_idx) > 0.1 * opt.numsample)
		mean_est = mean(bbox(selected_idx,:),1);

		x0 = min(X(selected_idx));
		y0 = min(Y(selected_idx));
		x1 = max(X(selected_idx));
		y1 = max(Y(selected_idx));
		%est_param = [min(x0, mean_est(1)), min(y0, mean_est(2)), max(x1-x0, mean_est(3)), max(y1-y0, mean_est(4))];
		if(x1-x0 < w_min)
			x0 = (x0+x1-w_min)/2;
			x1 = x0 + w_min;
		end
		if(y1-y0 < h_min)
			y0 = (y0+y1-h_min)/2;
			y1 = y0 + h_min;
		end
		est_param = [x0, y0, x1-x0, y1-y0];
	else
		if(isempty(selected_idx))
			[score, selected_idx] = sort(confidence, 1, 'descend');
			selected_idx = selected_idx(1:25);
		end
		est_param = mean(bbox(selected_idx,:),1);
	end
	
	% update window history
	if size(opt.window_hist, 1) < 4
		opt.window_hist = [opt.window_hist; [est_param(3), est_param(4)]];
	else
		opt.window_hist = [opt.window_hist(end-2:end, :); [est_param(3), est_param(4)]];
	end
	win_mean = mean(opt.window_hist, 1);
	est_param = [est_param(1) + 0.5 * (est_param(3) - win_mean(1)), est_param(2) + 0.5 * (est_param(4) - win_mean(2)), win_mean(1), win_mean(2)];
	
	if DEBUG
		not_selected_idx = setdiff([1:n]', selected_idx);
		hold on;		
		plot(X(not_selected_idx), Y(not_selected_idx), 'x', 'Color', 'c');
		plot(X(selected_idx), Y(selected_idx), 'x', 'Color', 'r');
		drawnow;
		pause(.1);
	end

	%est_param = mean(bbox(selected_idx,:),1);
	%{
	D = [X-(param.est(1)+param.est(3)/2), Y-(param.est(2)+param.est(4)/2)];
	mu = [0, 0];
	sigma = [opt.affsig(1), 0; 0, opt.affsig(2)];
	w = mvnpdf(D, mu, sigma);
	w = w ./ max(w);
	confidence = confidence .* w;
	
	conf = confidence - min(confidence);
	param.conf = exp(double(conf) ./opt.condenssig);
	param.conf = param.conf ./ sum(param.conf);
	[sorted_conf, sorted_idx] = sort(param.conf, 'descend');
	cumconf = cumsum(sorted_conf);
	idx = min(find(cumconf >= 0.8));
	selected_idx = sorted_idx(1:idx); % particles that within the bbox contains 98% energy
	%}

	param.est = est_param;
%end