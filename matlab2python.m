fn = 'texture4dxplus1dy0.mat';
im = load(fn)
%im.texture1dxmin1dymin1;
imshow(im.texture4dx1dy0)



%imtool(im.texture1dx1dymin1.mat);
imwrite(im.texture4dx1dy0, 'texture4dx1dy0.png')
%imwrite(im.g1d01, 'texture1dx1dy0.png')
%clear;