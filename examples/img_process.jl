using Images
using FFTW
using ParametricDFT

# load image
test_img = Images.load("examples/cat.png")
test_img = test_img[101:4:356,301:4:556]

# convert image to grayscale
function img2gray(img)
    return [Gray(img[i,j].g) for i in 1:size(img,1), j in 1:size(img,2)]
end
function img2mat(img)
    return [img[i,j].val for i in 1:size(img,1), j in 1:size(img,2)]
end

function mat2img(mat)
    return Gray.(mat)
end

# convert image to matrix
img_g = img2gray(test_img)
mat_g = img2mat(img_g)

# use fft in FFTW.jl to get the frequency domain
fftmat = fftshift(fft(mat_g))
fftabs = abs.(fftmat)
fftabs_normalized = fftabs ./ maximum(fftabs)
Gray.(fftabs_normalized)

# truncate the frequency domain
fftmat_truncate = copy(fftmat)
Gray.(ifft(ifftshift(fftmat)))

mid = 32
size = 10
fftmat_truncate[1:mid-size,:] .= 0
fftmat_truncate[mid+size:end,:] .= 0
fftmat_truncate[:,1:mid-size] .= 0
fftmat_truncate[:,mid+size:end] .= 0
pic_fft_truncate = Gray.(ifft(ifftshift(fftmat_truncate)))

# use ParametricDFT.jl to get the frequency domain
pic = vec(mat_g)
pic2 = reshape(mat_g,4096)

qubit_num = 12
# pic = rand(2^(qubit_num))
@time theta = ParametricDFT.fft_with_training(qubit_num, pic, ParametricDFT.L1Norm();steps = 200)
# steps = 1000: 2521.269052 seconds
# steps = 200: 531.288617 seconds

tensors = ParametricDFT.point2tensors(theta, qubit_num)
mat1 = ParametricDFT.ft_mat(tensors, ParametricDFT.qft_code(qubit_num)[1], qubit_num)

# save and load the matrix
using DelimitedFiles
# writedlm("examples/matrices/epoch200.txt", mat1)
mat1 = readdlm("examples/matrices/epoch200.txt",'\t',Complex{Float64})

Gray.(abs.(mat1)./ maximum(abs.(mat1)))

ft_pic = mat1*pic
reshape_ft_pic = reshape(ft_pic,64,64)
Gray.(abs.(reshape_ft_pic))

cut_threshold = 0.25
count(x->abs(x)<cut_threshold, ft_pic)
count(x->abs(x)>cut_threshold, ft_pic)

ft_pic_cut = copy(ft_pic)
ft_pic_cut[findall(x->abs(x)<cut_threshold, ft_pic)] .= 0

mat1'*ft_pic_cut 

pic_ParametricDFT_truncate = Gray.(reshape(mat1'*ft_pic_cut,64,64))

pic_fft_truncate
pic_ParametricDFT_truncate