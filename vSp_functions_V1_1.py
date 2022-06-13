## define functions
def loadImages(directory, file_list, sample, reduction, enchanced, dim, ench_type):
    ## Function takes file list and if samples --> loads sample, otherwise loads all images in file list, reducing by the reduction factor and enchancing by the enchance factor. If dim --> crops image otherwise entire image is loaded

    timeStamp = [] # create empty list for time stamps
    index = 0 # set index to zero, used as counter for assignemnt  to array


    if not sample:  # if sample does not exist create a list of numbers to load
        sample = range(len(file_list))  # make sample the entire file list

    if not dim:  # open an image and get the image dimensions if no cropping is supplied
        image = np.array(Image.open(directory+file_list[0]).reduce(int((reduction))))  # load image to array
        h, w, d = tuple(image.shape) # get shape of image in file
        dim = [0, h, 0, w,] # assign shape to dim list so entire image is imported

    # Create empty NP array for pixcels. All pictures are loaded into an array height, width, # images, colours
    pixcels = np.empty((dim[1] - dim[0], # number of rows
                        dim[3] - dim[2], # number of columns
                        len(sample), # number of images
                        3), # number of colour arrays per image + 3 for x and y and l
                       dtype='float32')

    for sam in sample:  # loop over each file in file_list
        clear_output(wait=True)  # clear output
        file = file_list[sam]  # set file as the sam'th entry in file list
        file_path = directory + file  # define file path
        ts = Image.open(file_path)._getexif()[36867]  # import timestamp as tS

        if enchanced:  # if an enchancments is call
            #open image and create enchancer
            if ench_type == 'Bright':
                enchancer = ImageEnhance.Brightness(Image.open(file_path).reduce(int((reduction))))
                image = np.array(enchancer.enhance(enchanced))  # load image and convert to array

            else:
                enchancer = ImageEnhance.Contrast(Image.open(file_path).reduce(int((reduction))))
                image = np.array(enchancer.enhance(enchanced))  # load image and convert to array

        else: # if no enchancement is called
            image = np.array(Image.open(file_path).reduce(int((reduction))))  # load image into array
        # crop array and load into pixcels array
        pixcels[:, :, index, :] = image[dim[0] : dim[1],
                                  dim[2] : dim[3],
                                  :]

        timeStamp.append(ts)  # append time stamp to list
        index += 1
        print("importing " + file + " file # " + str(sam + 1) + " of " + str(len(file_list) + 1))
        print(ts)
    return (timeStamp, pixcels)

def plotSamples(imgArr, ticks, title, titleData):
    ## function takes image array and plots images over 2 columns and as many rows as required. user selects if ticks and headings --> supply heading data in list form

    sampleSize = imgArr.shape[2]  # return the 3rd dimension of the array
    width = 2 # set number of columns to plot
    height = int(sampleSize/width) # determine how many rows to plot
    index = 0 # set a counter to 0
    plt.close()  # close previous plot
    f, axarr = plt.subplots(height, width)  # create enough rows to plot all samples
    for row in range(0,height):  # iterate over each image row
        for h in range(0,height): # iterare over each column
            axarr[h,row].imshow(imgArr[:, :, index, :].astype('uint8'))  # open image and plot
            if title:
                axarr[h, row].title.set_text(titleData[row])

            if not ticks:
                axarr[h, row].set_xticks([])
                axarr[h, row].set_yticks([])

            index += 1 # increase counter by 1

def clus2Image(clussArr, centers, recolour_dict):
    ## defien function to take an array of cluster labels and centers and return an image array, with specified clusters recoulured if required
    if recolour_dict: # if recolour provided
        for key in recolour_dict.keys(): # iterate over each key value pair
            centers[key] = np.asarray(recolour_dict[key]) # change the value of that center to the specified colour

    imgArr = centers[clussArr]

    return(imgArr)

def faissCluster(pixcels,frac, startClus, noIt):
    ## defien function to perfrom kmeans clustering using faiss
    no_clusters = startClus # set inital number of clusters
    no_iterations = noIt # set the no of iterations for each step of kmeans
    clusterMin = 1000 # set cluster min to enter while loop
    d = pixcels.shape[1]



    while clusterMin > frac:  # continue to increase number of clusters until the smallest cluster becomes sufficently small to just be the stripe
        #clear_output(wait=True)  # clear output
        print('clustering with '+str(no_clusters)+"\n")
        kmeans = faiss.Kmeans(d, no_clusters, niter=no_iterations, verbose=True) # define faiss kmeans object
        kmeans.train(pixcels) # train kmeans object
        D, I = kmeans.index.search(pixcels, 1) # return kmeans
        pixInClus = np.unique(I, return_counts=True) # get counts # pixels in each cluster
        #colour = np.where(pixInClus[1] == min(pixInClus[1])) # assigns the smallest cluster as the stripe colour
        colour = np.where(abs(pixInClus[1] / pixInClus[1].sum() - frac) == abs(pixInClus[1] / pixInClus[1].sum() - frac).min())
        clusterMin = min(pixInClus[1]) / (h*w*l) # Calcultes the fraction of the picture occupied the the stripeColour
        no_clusters += 1 # Increase number of clusters by 1
        print("Cluster Min was "+str(clusterMin)+"\n")

    return(D, I, kmeans, colour)

def cleanIQR(upper, lower, singleValues, inplace, verb):

    iqr = np.subtract(*np.percentile(singleValues, [upper, lower]))
    med = np.percentile(singleValues, 50)
    minus = med - iqr
    plus = med + iqr
    toKeep = (minus < singleValues) & (singleValues < plus)
    if verb:
        plt.close()
        f, axarr = plt.subplots(2)  # create enough rows to plot all samples
        axarr[0].boxplot(singleValues)
        axarr[1].boxplot(singleValues[(minus < singleValues) & (singleValues < plus)])
        print('The min projected y value before cleaning is '+str(singleValues.min()))
        print('The max projected y value before cleaning is '+str(singleValues.max()))
        print('The min projected y value after cleaning is '+
              str(singleValues[(minus < singleValues) & (singleValues < plus)].min()))
        print('The max projected y value after cleaning is '+
              str(singleValues[(minus < singleValues) & (singleValues < plus)].max()))

    if inplace:
        return(singleValues[(minus < singleValues) & (singleValues < plus)])
    else:
        return(toKeep)

def addInd(w,h,l, pixcels, Vh):
    ## define function to add x and y cordinated, projected onto second principle component of the stripe to the pixcels array so as geometric position has an effect
    i, j, p = np.meshgrid(range(w),range(h),range(l)) # use mesh grid to create arrays of indices
    i= np.reshape(i, (w*h*l,1)).astype('float32') # reshape x ind to match reshaped pixcels
    j= np.reshape(j, (w*h*l,1)).astype('float32') # reshape y ind to suit reshaped pixcels
    searchArray = np.hstack((j,i)) # stack ontop
    del i
    del j
    del p
    if Vh: # only if using Vh
        searchArray = searchArray @Vh[:,1] # multiply by rotation vector
        plt.close()
        plt.boxplot(searchArray) # plot to check range
        searchArray = np.interp(searchArray, (searchArray.min(), searchArray.max()), (-1,1))
        print('scaled min of dot_search is '+str(searchArray.min()))
        print('scaled max of dot_search is '+str(searchArray.max()))

    else: # if no rotation matrix is specified note - this carries both x and y through. if wanting to drop x simply defien a straight rotation matrix
        plt.close()
        plt.boxplot(searchArray) # plot to check range
        searchArray = np.interp(searchArray, (searchArray.min(), searchArray.max()), (-1,1))
        print('scaled min of dot_search is '+str(searchArray.min()))
        print('scaled max of dot_search is '+str(searchArray.max()))
    searchArray = np.reshape(searchArray,(h*w*l,1))
    searchArray= np.hstack((pixcels,searchArray))
    ## where is the y component dropped if not using Vh make it carry through
    return(searchArray.astype('float32'))

def GetStats(stripe_loc, perpClean_min, perpClean_max, rotClean_min, rotClean_max):
    pca = PCA(2) # define scikitLearn PCA function
    pca.fit(stripe_loc[0:2,:].T) # perform PCA on the x and y componets of the stripe loctions
    Vh = pca.components_ # extract the rotation matrix
    stripe_perp = stripe_loc[0:2,:].T @ Vh[:,1] # rotate stripe onto second principle component of stripe to clean
    stripe_perp = cleanIQR(perpClean_max,perpClean_min,stripe_perp, False, False) # remove outliers from stripe rot
    stripe_rot = stripe_loc[0:2,:].T @ Vh[:,0] # rotate stripe onto first principle component
    stripe_rot = stripe_rot[stripe_perp] # remove values which are removed in the perp cleaning process
    stripe_rot = cleanIQR(rotClean_max,rotClean_min,stripe_rot, True, False)
    ## do i need to shape stripe_perp? and stripe Rot to subset stripe Loc
    stripe_loc = stripe_loc[stripe_perp] # remove perp outliers from stripe loc
    stripe_loc = stripe_loc[stripe_rot] # remove rot outliers from loc
    stripeMedian = np.median(stripe_rot) # find median of the stripe
    stripe_len = (abs(abs(np.min(stripe_rot)) - abs(np.max(stripe_rot)))) # finds the abs range of the stripe (known length)

    return(stripeMedian, stripe_len, Vh, stripe_loc)