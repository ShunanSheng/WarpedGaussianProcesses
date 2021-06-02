clear all;
clc;
A=diag(ones(2,1));
func = @(x, y)(quadratic_form(x, y, A));
integral_value = integral2(func, -1,1,-1,1)

function value = quadratic_form(x, y, A)
    output_dim=size(x);
    x=reshape(x,[],1);
    y=reshape(y,[],1);
    mat = [x,y];
%     value = diag(mat * A * mat');
    value= sum((mat*A).*mat,2); % A better implementation to find sum(x'Ax), 
    value=reshape(value,output_dim(1),output_dim(2));
end

