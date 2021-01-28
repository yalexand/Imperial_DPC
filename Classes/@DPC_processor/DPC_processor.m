classdef DPC_processor < handle 
       
    properties(Constant)
        %
    end
    
    properties(SetObservable = true)                                                                                   
    end                    
    
    properties(Transient) 
        
    zernike_poly  = [];
    pupil = [];
    dim = [];
    source = [];
    
    % direction convention 1-down, 2-up, 3-left, 4-right
    IDPC = [];
    fIDPC = []; % Fourier
    
    pupilphase = [];
    f_amplitude = [];
    f_phase = [];
    %
    use_gpu = false;
    %
    % used in iterative aberrations correction
    padsize = [];
    Dx = [];
    Dy = [];
        
    na            = [];   % numerical aperture of the imaging system
    na_illum      = [];   % numerical aperture of the illumination
    magnification = [];   % magnification of the imaging system
    lambda        = [];   % wavelength in micron
    ps            = [];   % pixel size in micron
    
    illu_rotation = [];

    % if annular illumination is used, set the na corresponds to the inner radius     
    na_inner      = [0, 0, 0, 0];                   
    
    num_Zernike   = 21;           
        
    x = [];
    y = [];
    fx = [];
    fy = [];
    Fx = [];
    Fy = [];
    
    end    
        
    properties(Transient,Hidden)        
        run_headless = false;        
    end
    
    events        
    end
            
    methods
%-------------------------------------------------------------------------%        
        function obj = DPC_processor()            
        end                
%-------------------------------------------------------------------------%                
function load_images(obj,fullfilename,permute_flag,~)
    [FILEPATH,NAME,EXT] = fileparts(fullfilename);
    if contains(EXT,'mat')
        load(fullfilename);
        if permute_flag
            IDPC = permute(double(IDPC), [2, 3, 1]);
        end
        for image_index = 1:size(IDPC, 3)
            image_load = IDPC(:, :, image_index);
            IDPC(:, :, image_index) = image_load; 
        end
        obj.dim = [size(IDPC, 1), size(IDPC, 2)];
        obj.IDPC = IDPC;
        clear('IDPC');
    else
        %[~,~,I] = bfopen(fullfilename); % TO DO 
    end
end
%-------------------------------------------------------------------------%                
function normalize_images(obj,bckg_filename,offset,~)
    
    if isempty(bckg_filename)
        for k=1:size(obj.IDPC,3)
            u = obj.IDPC(:,:,k);
            obj.IDPC(:,:,k) = u/mean2(u)-1;
        end
    else % this doesn't help
        %[~,~,b] = bfopen_v(bckg_filename);
        for k=1:size(obj.IDPC,3)
            u = obj.IDPC(:,:,k);
            obj.IDPC(:,:,k) = u/mean2(u)-1;
        end        
    end        
end
%-------------------------------------------------------------------------%
function load_image(obj,fullfilename,index,~)
    [~,~,I] = bfopen_v(fullfilename);
    obj.dim = [size(I,1),size(I,2)];
    if isempty(obj.IDPC)
        obj.IDPC = zeros(obj.dim(1),obj.dim(2));
    end
    obj.IDPC(:,:,index) = single(I);
    %icy_imshow(I,num2str(index));
end
%-------------------------------------------------------------------------%
function setCoordinates(obj,~,~)
    dim = obj.dim;
    ps = obj.ps;
    % set coordinates x and y in real space
    x        = -(dim(2)-mod(dim(2), 2))/2:1:(dim(2)-mod(dim(2), 2))/2-(mod(dim(2), 2)==0);
    x        = ps*x;
    y        = -(dim(1)-mod(dim(1), 2))/2:1:(dim(1)-mod(dim(1), 2))/2-(mod(dim(1), 2)==0);
    y        = ps*y;

    % set coordinates fx and fy in Fourier space
    dfx      = 1/dim(2)/ps; dfy = 1/dim(1)/ps;
    fx       = -(dim(2)-mod(dim(2), 2))/2:1:(dim(2)-mod(dim(2), 2))/2-(mod(dim(2), 2)==0);
    fx       = dfx*fx;
    fy       = -(dim(1)-mod(dim(1), 2))/2:1:(dim(1)-mod(dim(1), 2))/2-(mod(dim(1), 2)==0);
    fy       = dfy*fy;
    fx       = ifftshift(fx);
    fy       = ifftshift(fy);

    % generate 2D grid
    [Fx, Fy] = meshgrid(fx, fy);

    obj.x=x;
    obj.y=y;
    obj.fx=fx;
    obj.fy=fy;
    obj.Fx=Fx;
    obj.Fy=Fy;
end
%-------------------------------------------------------------------------%
function genSource_no_aberration_correction(obj,~,~)
    
    for k = 1:numel(obj.illu_rotation) 
        S0   = (sqrt(obj.Fx.^2+obj.Fy.^2)*obj.lambda<=obj.na_illum & sqrt(obj.Fx.^2+obj.Fy.^2)*obj.lambda>=obj.na_inner(k)*obj.na_illum);
        mask = zeros(size(obj.Fx));
        %        
        % asymmetric mask based on illumination angle
        if obj.illu_rotation(k) < 180 || obj.illu_rotation(k) == 270 
            mask(obj.Fy>=(obj.Fx*tand(obj.illu_rotation(k)))) = 1;
        else
            mask(obj.Fy<=(obj.Fx*tand(obj.illu_rotation(k)))) = 1;    
        end
        obj.source(:,:,k) = S0.*mask;
        %icy_imshow(fftshift(S0.*mask),[num2str(k) ' ' num2str(obj.na_inner(k))]);
    end
end
%-------------------------------------------------------------------------%
function generate_pupil(obj,~,~)
    obj.pupil = (obj.Fx.^2+obj.Fy.^2<=(obj.na/obj.lambda)^2);    
end
%-------------------------------------------------------------------------%
function [amplitude, phase] = DPC_L2(obj,reg_L2,~)    
    % initalization of Zernike coefficients for pupil estimation, ignoring the first three orders    
    zernike_coeff     = 0*randn(obj.num_Zernike-3, 1);   
        
    F             = @(x) fft2(x);
    IF            = @(x) ifft2(x);    

    % calculate frequency spectrum of the measurements
    if isempty(obj.fIDPC)
        obj.fIDPC = F(obj.IDPC);    
    end

    fIDPC = obj.fIDPC;
    
    if obj.use_gpu
        % place measurements and variables into GPU memory
        source = gpuArray(obj.source);    
        pupil  = gpuArray(obj.pupil);
        fIDPC  = gpuArray(obj.fIDPC);
    else
        source = obj.source;    
        pupil  = obj.pupil;
    end    
    
    Hi              = zeros(obj.dim(1), obj.dim(2), size(source, 3));
    Hr              = zeros(obj.dim(1), obj.dim(2), size(source, 3));    
    if obj.use_gpu
        Hi = gpuArray(Hi);
        Hr = gpuArray(Hr);
    end
    
    pupilphase      = obj.aberrationGeneration(zernike_coeff);
    pupil_aberrated = pupil.*exp(1i*pupilphase);

    for source_index = 1:size(source, 3)
        [Hi(:, :, source_index),...
         Hr(:, :, source_index)] = genTransferFunction(source(:, :, source_index), pupil_aberrated);
    end
    
    % matrix pseudo inverse
    M11             = sum(abs(Hr).^2, 3) + reg_L2(1);
    M12             = sum(conj(Hr).*Hi, 3);
    M21             = sum(conj(Hi).*Hr, 3);
    M22             = sum(abs(Hi).^2, 3) + reg_L2(2);
    denominator     = M11.*M22 - M12.*M21;
    I1              = sum(fIDPC.*conj(Hr), 3);
    I2              = sum(fIDPC.*conj(Hi), 3);
    amplitude       = real(IF((I1.*M22-I2.*M12)./denominator));
    phase           = real(IF((I2.*M11-I1.*M21)./denominator));
    
    if(obj.use_gpu)
        amplitude = gather(amplitude);
        phase = gather(phase);
    end    
end
%-------------------------------------------------------------------------%
function aberration = aberrationGeneration(obj,zernike_coeff,~)
    % global zernike_poly dim;
    aberration = obj.zernike_poly*zernike_coeff;
    aberration = reshape(aberration,obj.dim);
end
%-------------------------------------------------------------------------%
function [amplitude_k,phase_k] = restore_with_aberrations_correction(obj,max_iter_algorithm,use_tv,verbose,~)
    
    F             = @(x) fft2(x);
    
    obj.setCoordinates;
    obj.genSource_aberration_correction;
    obj.zernike_poly = genZernikePoly(obj.Fx, obj.Fy, obj.na, obj.lambda, obj.num_Zernike);        
    obj.generate_pupil;    
     
    tau                 = [1e-5, 5e-3];                % parameters for total variation [amplitude, phase] (can set to a very small value in noiseless case)
    reg_L2              = 1.0*[1e-1, 5e-3];            % parameters for L2 regurlarization [amplitude, phase] (can set to a very small value in noiseless case)    
    
    options.Method      = 'lbfgs';
    options.maxIter     = 10;
    options.PROGTOL     = 1e-30;
    options.optTol      = 1e-30;
    options.MAXFUNEVALS = 500;
    options.corr        = 50;
    options.usemex      = 0;
    options.display     = false;
    
    %
    zernike_coeff_k     = 0*randn(obj.num_Zernike-3, 1);   % initalization of Zernike coefficients for pupil estimation, ignoring the first three orders
    
    obj.fIDPC = F(obj.IDPC);
    
    t_start = tic;
    for iter = 1:max_iter_algorithm
        if ~use_tv
            % Least-Squares with L2 regularization
            [amplitude_k, phase_k] = obj.DPC_L2(reg_L2);            
        else
        % ADMM algorithm with total variation regularization
        %global padsize Dx Dy;
        obj.padsize                      = 0;
        temp                         = zeros(obj.dim);
        temp(1, 1)                   = 1;
        temp(1, end)                 = -1;
        obj.Dx                           = F(temp);
        temp                         = zeros(obj.dim);
        temp(1, 1)                   = 1;
        temp(end, 1)                 = -1;
        obj.Dy                           = F(temp);
        rho                          = 1;
        D_x                          = zeros(obj.dim(1), obj.dim(2), 4);
        u_k                          = zeros(obj.dim(1), obj.dim(2), 4);
        z_k                          = zeros(obj.dim(1), obj.dim(2), 4);
        if obj.use_gpu
           obj.Dx = gpuArray(obj.Dx);
           obj.Dy = gpuArray(obj.Dy);
           D_x= gpuArray(D_x);
           u_k= gpuArray(u_k);
           z_k= gpuArray(z_k);
        end
        
        for iter_ADMM = 1:20
           [amplitude_k, phase_k] = obj.DPC_TV(zernike_coeff_k, rho, z_k, u_k, reg_L2);
            if iter_ADMM < 20
                D_x(:, :, 1)   = amplitude_k - circshift(amplitude_k, [0, -1]);
                D_x(:, :, 2)   = amplitude_k - circshift(amplitude_k, [-1, 0]);
                D_x(:, :, 3)   = phase_k - circshift(phase_k, [0, -1]);
                D_x(:, :, 4)   = phase_k - circshift(phase_k, [-1, 0]);
                z_k            = D_x + u_k;
                z_k(:, :, 1:2) = max(z_k(:, :, 1:2) - tau(1)/rho, 0) -...
                                 max(-z_k(:, :, 1:2) - tau(1)/rho, 0);
                z_k(:, :, 3:4) = max(z_k(:, :, 3:4) - tau(2)/rho, 0) -...
                                 max(-z_k(:, :, 3:4) - tau(2)/rho, 0);
                u_k            = u_k + (D_x-z_k);
            end
        end
            clear u_k z_k D_x;
        end
            
        obj.f_amplitude = F(amplitude_k);
        obj.f_phase     = F(phase_k);

        % pupil estimation
        [zernike_coeff_k, loss(iter)] = minFunc(@obj.gradientPupil, zernike_coeff_k, options);
        %pupilphase                    = aberrationGeneration(zernike_coeff_k);
        
        % print cost function value and computation time at each iteration
        if verbose
            fprintf('iteration: %04d, loss: %5.5e, elapsed time: %4.2f seconds\n', iter, loss(iter), toc(t_start));   
        end                                                
    end 
    
    if obj.use_gpu
        amplitude_k = gather(amplitude_k);
        phase_k = gather(phase_k);
    end
end
%-------------------------------------------------------------------------%
function [amplitude, phase] = DPC_TV(obj,zernike_coeff, rho, z_k, u_k, reg_L2,~)
    
    F             = @(x) fft2(x);
    IF            = @(x) ifft2(x);    
    % calculate frequency spectrum of the measurements
    obj.fIDPC = F(obj.IDPC);        
    
    Hi              = zeros(obj.dim(1), obj.dim(2), size(obj.source, 3));
    Hr              = zeros(obj.dim(1), obj.dim(2), size(obj.source, 3));
    if obj.use_gpu
        Hi = gpuArray(Hi);
        Hr = gpuArray(Hr);
    end
    pupilphase      = obj.aberrationGeneration(zernike_coeff);
    pupil_aberrated = obj.pupil.*exp(1i*pupilphase);

    for source_index = 1:size(obj.source, 3)
        [Hi(:, :, source_index),...
         Hr(:, :, source_index)] = genTransferFunction(obj.source(:, :, source_index), pupil_aberrated);
    end
    
    M11         = sum(abs(Hr).^2, 3) + rho * abs(obj.Dx).^2 + rho * abs(obj.Dy).^2 + reg_L2(1);
    M12         = sum(conj(Hr).*Hi, 3);
    M21         = sum(conj(Hi).*Hr, 3);
    M22         = sum(abs(Hi).^2, 3) + rho * abs(obj.Dx).^2 + rho * abs(obj.Dy).^2 + reg_L2(2);
    denominator = M11.*M22-M12.*M21;
    b2          = F(padarray(z_k - u_k, [obj.padsize/2, obj.padsize/2,0]));
    I1          = sum(obj.fIDPC.*conj(Hr), 3) + rho*(conj(obj.Dx).*b2(:, :, 1) + conj(obj.Dy).*b2(:, :, 2));
    I2          = sum(obj.fIDPC.*conj(Hi), 3) + rho*(conj(obj.Dx).*b2(:, :, 3) + conj(obj.Dy).*b2(:, :, 4));
    phase       = real(IF((I2.*M11-I1.*M21)./(denominator+eps)));
    amplitude   = real(IF((I1.*M22-I2.*M12)./(denominator+eps)));
    phase       = phase(obj.padsize/2+1:end-obj.padsize/2, obj.padsize/2+1:end-obj.padsize/2);
    amplitude   = amplitude(obj.padsize/2+1:end-obj.padsize/2, obj.padsize/2+1:end-obj.padsize/2);
end
%-------------------------------------------------------------------------%
function genSource_aberration_correction(obj,~,~)        
    obj.genSource_no_aberration_correction;
    % replace last image by single point source
        point_source = zeros(obj.dim(1), obj.dim(2));
        point_source(obj.Fx==0 & obj.Fy==0) = 1;  
    obj.source = cat(3,obj.source,point_source);
end
%-------------------------------------------------------------------------%
function [f, g] = gradientPupil(obj,zernike_est,~)
    % global pupil source fIDPC f_amplitude f_phase use_gpu;
    F           = @(x) fft2(x);
    IF          = @(x) ifft2(x);
    if obj.use_gpu
        zernike_est = gpuArray(zernike_est);
    end
    f           = 0;
    g           = 0;
    pupil_phase = obj.aberrationGeneration(zernike_est);
    pupil_est   = obj.pupil.*exp(1i*pupil_phase);
    
    for source_index = 1:size(obj.source, 3)
        % forward model
        source_f      = rot90(padarray(fftshift(obj.source(:, :, source_index)), [1, 1], 'post'), 2);
        source_f      = ifftshift(source_f(1:end-1, 1:end-1));
        DC            = sum(sum(source_f.*abs(pupil_est).^2));
        f_sp          = F(source_f.*pupil_est);
        f_p           = F(pupil_est);
        H_first_half  = conj(f_sp).*f_p;
        H_second_half = conj(f_p).*f_sp;
        
        % compute cost function value
        residual      = obj.fIDPC(:,:,source_index) -...
                        (IF(H_first_half+H_second_half).*obj.f_amplitude +...
                        1i*IF(H_first_half-H_second_half).*obj.f_phase)/DC;
        f             = f + 0.5*norm(residual(:))^2;
        
        % compute gradient
        backprop_1    = F(conj(obj.f_amplitude).*residual);
        backprop_2    = F(-1i*conj(obj.f_phase).*residual);
        grad_pupil    = (IF(f_sp.*(backprop_1+backprop_2)) +...
                        source_f.*IF(f_p.*(backprop_1-backprop_2)))/DC;
        g             = g - obj.aberrationDecomposition(-1i*conj(pupil_est).*grad_pupil);
    end
    f = gather(f);
    g = gather(real(g)); 
end
%-------------------------------------------------------------------------%
function zernike_coeff = aberrationDecomposition(obj,aberration,~)
    zernike_coeff = obj.zernike_poly'*aberration(:);
end
%-------------------------------------------------------------------------%
    end % methods
end % classdef
