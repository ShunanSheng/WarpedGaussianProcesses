function value = test_integrand(epsilon_cell, A, z, vhat, sig)

output_size = size(epsilon_cell{1})

dim = length(epsilon_cell);

epsilon_mat = zeros(prod(output_size), dim);

for d = 1:dim
    epsilon_mat(:, d) = epsilon_cell{d}(:);
end

value = exp(-0.5 * sum(epsilon_mat .^ 2, 2) / (sig ^ 2));


vec = -epsilon_mat + z' - vhat';
qform = sum((vec * A) .* vec, 2);

value = value .* exp(-0.5 * qform);

value = reshape(value, output_size(1), output_size(2));

end

