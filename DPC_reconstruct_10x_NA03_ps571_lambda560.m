function phase = DPC_reconstruct_10x_NA03_ps571_lambda560(Top,Bottom,Left,Right)

% presumed it starts from root
addpath(pwd);
addpath_DPC;

% SET UP RECONSTRUCTION ENGINE
    dpc = DPC_processor;
    dpc.use_gpu = true;    
%    
%     10x example
    dpc.na            = 0.3;                       % numerical aperture of the imaging system
    dpc.na_illum      = 0.3;                       % numerical aperture of the illumination
    dpc.magnification = 10;                        % magnification of the imaging system
    dpc.ps            = 0.571;                     % pixel size in micron    
    dpc.lambda        = 0.560;                     % wavelength in micron
    dpc.illu_rotation = [0, 180, 90, 270];   
    %
    reg_L2 = 1.0*[1e-1, 5e-3]; % regularization coeffs
    
    [sx,sy] = size(Top);
    
    dpc.IDPC = zeros(sx,sy,4);
    dpc.IDPC(:,:,1) = Top;
    dpc.IDPC(:,:,2) = Bottom;
    dpc.IDPC(:,:,3) = Left;
    dpc.IDPC(:,:,4) = Right;
    
    dpc.dim = [sx,sy];
                
        disp('Performing DPC reconstruction .. ');
        
                dpc.setCoordinates;
                dpc.genSource_no_aberration_correction;
                dpc.zernike_poly = genZernikePoly(dpc.Fx, dpc.Fy, dpc.na, dpc.lambda, dpc.num_Zernike);        
                dpc.generate_pupil;
                dpc.normalize_images([],[]);    
            %            
            t0=tic;                        
                    [~,phase] = dpc.DPC_L2(reg_L2);
            toc(t0)  
            
            %figure();
            %phase_show = uint8(255*phase-min(phase(:))./(max(phase(:))-min(phase(:))));
            %imshow(phase_show);
            phase = single(phase);
end
