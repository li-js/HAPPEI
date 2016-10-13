function rect2=rectify_rect(rect, im_sz, enlarge)
    if nargin==1
        enlarge=1;
    end
    
    x_c=round(rect(1)+0.5*rect(3));
    y_c=round(rect(2)+0.5*rect(4));
    
    half=round(0.5*enlarge*max(rect(3:4)));
    upper=min([x_c, im_sz(2)-x_c, y_c, im_sz(1)-y_c]);
    half=min([half, upper]);
    
    rect2=[x_c-half, y_c-half, 2*half, 2*half];
end