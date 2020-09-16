function restruct(input,num)
    figure();
    subplot(131);
    imshow(input,[]);
    title('original image');
    
    % get sinogram
    angle=linspace(0,180,num);
    R=radon(input,angle);
    subplot(132);
    imshow(R,[]);
    title('sinogram');
    
    % restruct
    result=iradon(R,angle);
    subplot(133);
    imshow(result,[]);
    title('restructed image');
end