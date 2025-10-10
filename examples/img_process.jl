using Images
using FFTW
using QDFT

test_img = Images.load("examples/cat.png")
test_img = test_img[101:4:356,301:4:556]

function img2gray(img)
    return [Gray(img[i,j].g) for i in 1:size(img,1), j in 1:size(img,2)]
end
function img2mat(img)
    return [img[i,j].val for i in 1:size(img,1), j in 1:size(img,2)]
end

function mat2img(mat)
    return Gray.(mat)
end
img_g = img2gray(test_img)
mat_g = img2mat(img_g)

fftmat = fftshift(fft(mat_g))
fftabs = abs.(fftmat)
fftabs_normalized = fftabs ./ maximum(fftabs)
Gray.(fftabs_normalized)

fftmat_truncate = copy(fftmat)

Gray.(ifft(ifftshift(fftmat)))

mid = 32
size = 10
fftmat_truncate[1:mid-size,:] .= 0
fftmat_truncate[mid+size:end,:] .= 0
fftmat_truncate[:,1:mid-size] .= 0
fftmat_truncate[:,mid+size:end] .= 0
Gray.(ifft(ifftshift(fftmat_truncate)))


pic = vec(mat_g)

qubit_num = 12
# pic = rand(2^(qubit_num))
theta = QDFT.fft_with_training(qubit_num, pic, QDFT.L1Norm())
@show theta