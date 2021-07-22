import numpy as np
import readimage as ri
# import calculatedistance as cd
np.set_printoptions(threshold = np.inf)

readimageInstance = ri.ReadImage()
spectrumImageArray = readimageInstance.funReadMat('/Users/didongchang/Desktop/Hyperspectral_dataset/Salinas/Salinas_scene/salinas_corrected.mat')
reshapedImageArray = spectrumImageArray.reshape(-1,spectrumImageArray.shape[-1])
groundTruth = readimageInstance.funReadMat('/Users/didongchang/Desktop/Hyperspectral_dataset/Salinas/Salinas_scene/salinas_gt.mat')
reshapedgroundTruth = groundTruth.reshape((-1,1))

imageArrayWithLable = []
for i in range(groundTruth.shape[0]):
	for j in range(groundTruth.shape[1]):
		if groundTruth[i][j] != 0:
			templist = [reshapedImageArray[i][j],groundTruth[i][j]] 
			templist.append(groundTruth[i][j])
			imageArrayWithLable.append(templist)

print(imageArrayWithLable)
print(len(imageArrayWithLable))
