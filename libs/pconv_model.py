import os
import sys
import time
from datetime import datetime
import logging
import traceback
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # For "OSError: image file is truncated"

sys.path.append(os.pardir)

import const as cst

from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, UpSampling2D, Dropout, LeakyReLU, BatchNormalization, Activation
from keras.layers.merge import Concatenate
from keras.applications import VGG16
from keras import backend as K
from libs.pconv_layer import PConv2D
import keras
import tensorflow as tf


class PConvUnet(object):
    
    # training: Ues 'CROP_HEIGHT'
    # predict: Use 'MAX_HEIGHT'
    def __init__(self, img_rows=cst.MAX_HEIGHT, img_cols=cst.MAX_WIDTH, weight_filepath=None):
        """Create the PConvUnet. If variable image size, set img_rows and img_cols to None"""
        
        # Settings
        self.weight_filepath = weight_filepath
        self.img_rows = img_rows
        self.img_cols = img_cols
        assert self.img_rows >= img_rows, 'Height must be >=256 pixels'
        assert self.img_cols >= img_cols, 'Width must be >=256 pixels'

        # Set current epoch
        self.current_epoch = 0
        
        # VGG layers to extract features from (first maxpooling layers, see pp. 7 of paper)
        self.vgg_layers = [3, 6, 10]
        
        # Get the vgg16 model for perceptual loss        
        self.vgg = self.build_vgg()
        
        # Create UNet-like model
#         self.model = self.build_pconv_unet(lr = 0.0002 * 0.1)
        self.model = self.build_pconv_unet()
        

        
    def build_vgg(self):
        """
        Load pre-trained VGG16 from keras applications
        Extract features to be used in loss function from last conv layer, see architecture at:
        https://github.com/keras-team/keras/blob/master/keras/applications/vgg16.py
        """
        
#         with tf.device("/cpu:0"):
#             # Input image to extract features from
#             img = Input(shape=(self.img_rows, self.img_cols, 3))

#             # Get the vgg network from Keras applications
#             vgg = VGG16(weights="imagenet", include_top=False)
    
#             # Output the first three pooling layers
#             vgg.outputs = [vgg.layers[i].output for i in self.vgg_layers]
    
#             # Create model and compile
#             cpu_model = Model(inputs=img, outputs=vgg(img))            
            
#         model = cpu_model#keras.utils.multi_gpu_model(cpu_model, gpus=4)
#         model.trainable = False
                    
#         model.compile(loss='mse', optimizer='adam')
        
#         return model


        img = Input(shape=(self.img_rows, self.img_cols, 3))
        vgg = VGG16(weights="imagenet", include_top=False)
        vgg.outputs = [vgg.layers[i].output for i in self.vgg_layers]
        model = Model(inputs=img, outputs=vgg(img))
        model.trainable = False
        model.compile(loss='mse', optimizer='adam')
        
        return model
        
    def build_pconv_unet(self, train_bn=True, lr=0.0002):      
        # INPUTS
        inputs_img = Input((self.img_rows, self.img_cols, 3))
        inputs_mask = Input((self.img_rows, self.img_cols, 3))
        
        # ENCODER
        def encoder_layer(img_in, mask_in, filters, kernel_size, bn=True):
            conv, mask = PConv2D(filters, kernel_size, strides=2, padding='same')([img_in, mask_in])
            if bn:
                conv = BatchNormalization(name='EncBN'+str(encoder_layer.counter))(conv, training=train_bn)
            conv = Activation('relu')(conv)
            encoder_layer.counter += 1
            return conv, mask
        
        # DECODER
        def decoder_layer(img_in, mask_in, e_conv, e_mask, filters, kernel_size, bn=True):
            up_img = UpSampling2D(size=(2,2))(img_in)
            up_mask = UpSampling2D(size=(2,2))(mask_in)
            concat_img = Concatenate(axis=3)([e_conv,up_img])
            concat_mask = Concatenate(axis=3)([e_mask,up_mask])
            conv, mask = PConv2D(filters, kernel_size, padding='same')([concat_img, concat_mask])
            if bn:
                conv = BatchNormalization()(conv)
            conv = LeakyReLU(alpha=0.2)(conv)
            return conv, mask
        
        
        # Setup the model inputs / outputs
#         with tf.device("/cpu:0"):
#             # INPUTS
#             inputs_img = Input((self.img_rows, self.img_cols, 3))
#             inputs_mask = Input((self.img_rows, self.img_cols, 3))
            
#             encoder_layer.counter = 0
        
#             e_conv1, e_mask1 = encoder_layer(inputs_img, inputs_mask, 64, 7, bn=False) # 64
#             e_conv2, e_mask2 = encoder_layer(e_conv1, e_mask1, 128, 5) # 128
#             e_conv3, e_mask3 = encoder_layer(e_conv2, e_mask2, 256, 5) # 256
#             e_conv4, e_mask4 = encoder_layer(e_conv3, e_mask3, 256, 3) # 256
#             e_conv5, e_mask5 = encoder_layer(e_conv4, e_mask4, 256, 3) # 256
#             e_conv6, e_mask6 = encoder_layer(e_conv5, e_mask5, 256, 3) # 256
#             e_conv7, e_mask7 = encoder_layer(e_conv6, e_mask6, 256, 3) # 256
#             e_conv8, e_mask8 = encoder_layer(e_conv7, e_mask7, 256, 3) # 256
            
#             d_conv9, d_mask9 = decoder_layer(e_conv8, e_mask8, e_conv7, e_mask7, 256, 3) # 256
#             d_conv10, d_mask10 = decoder_layer(d_conv9, d_mask9, e_conv6, e_mask6, 256, 3) # 256
#             d_conv11, d_mask11 = decoder_layer(d_conv10, d_mask10, e_conv5, e_mask5, 256, 3) # 256
#             d_conv12, d_mask12 = decoder_layer(d_conv11, d_mask11, e_conv4, e_mask4, 256, 3) # 256
#             d_conv13, d_mask13 = decoder_layer(d_conv12, d_mask12, e_conv3, e_mask3, 256, 3) # 256
#             d_conv14, d_mask14 = decoder_layer(d_conv13, d_mask13, e_conv2, e_mask2, 128, 3) # 128
#             d_conv15, d_mask15 = decoder_layer(d_conv14, d_mask14, e_conv1, e_mask1, 64, 3)
#             d_conv16, d_mask16 = decoder_layer(d_conv15, d_mask15, inputs_img, inputs_mask, 3, 3, bn=False)
#             outputs = Conv2D(3, 1, activation = 'sigmoid')(d_conv16)        
        
#             cpu_model = Model(inputs=[inputs_img, inputs_mask], outputs=outputs)
            
#         model = keras.utils.multi_gpu_model(cpu_model, gpus=4)

        encoder_layer.counter = 0

        e_conv1, e_mask1 = encoder_layer(inputs_img, inputs_mask, 64, 7, bn=False)
        e_conv2, e_mask2 = encoder_layer(e_conv1, e_mask1, 128, 5)
        e_conv3, e_mask3 = encoder_layer(e_conv2, e_mask2, 256, 5)
        e_conv4, e_mask4 = encoder_layer(e_conv3, e_mask3, 256, 3)
        e_conv5, e_mask5 = encoder_layer(e_conv4, e_mask4, 256, 3)
        e_conv6, e_mask6 = encoder_layer(e_conv5, e_mask5, 256, 3)
        e_conv7, e_mask7 = encoder_layer(e_conv6, e_mask6, 256, 3)
        e_conv8, e_mask8 = encoder_layer(e_conv7, e_mask7, 256, 3)
        
        d_conv9, d_mask9 = decoder_layer(e_conv8, e_mask8, e_conv7, e_mask7, 256, 3)
        d_conv10, d_mask10 = decoder_layer(d_conv9, d_mask9, e_conv6, e_mask6, 256, 3)
        d_conv11, d_mask11 = decoder_layer(d_conv10, d_mask10, e_conv5, e_mask5, 256, 3)
        d_conv12, d_mask12 = decoder_layer(d_conv11, d_mask11, e_conv4, e_mask4, 256, 3)
        d_conv13, d_mask13 = decoder_layer(d_conv12, d_mask12, e_conv3, e_mask3, 256, 3)
        d_conv14, d_mask14 = decoder_layer(d_conv13, d_mask13, e_conv2, e_mask2, 128, 3)
        d_conv15, d_mask15 = decoder_layer(d_conv14, d_mask14, e_conv1, e_mask1, 64, 3)
        d_conv16, d_mask16 = decoder_layer(d_conv15, d_mask15, inputs_img, inputs_mask, 3, 3, bn=False)
        outputs = Conv2D(3, 1, activation = 'sigmoid')(d_conv16)        
        
        # Setup the model inputs / outputs
        model = Model(inputs=[inputs_img, inputs_mask], outputs=outputs)
        
        
        # Compile the model
        model.compile(
            optimizer = Adam(lr=lr),
            loss=self.loss_total(inputs_mask)
        )

        return model
    
    
    def loss_total(self, mask):
        """
        Creates a loss function which sums all the loss components 
        and multiplies by their weights. See paper eq. 7.
        """
        def loss(y_true, y_pred):
            
            # Compute predicted image with non-hole pixels set to ground truth
            y_comp = mask * y_true + (1-mask) * y_pred
            
            # Compute the vgg features
            vgg_out = self.vgg(y_pred)
            vgg_gt = self.vgg(y_true)
            vgg_comp = self.vgg(y_comp)
            
            # Compute loss components
            l1 = self.loss_valid(mask, y_true, y_pred)
            l2 = self.loss_hole(mask, y_true, y_pred)
            l3 = self.loss_perceptual(vgg_out, vgg_gt, vgg_comp)
            l4 = self.loss_style(vgg_out, vgg_gt)
            l5 = self.loss_style(vgg_comp, vgg_gt)
            l6 = self.loss_tv(mask, y_comp)
            
            # Return loss function
            return l1 + 6*l2 + 0.05*l3 + 120*(l4+l5) + 0.1*l6

        return loss
    
    def loss_hole(self, mask, y_true, y_pred):
        """Pixel L1 loss within the hole / mask"""
        return self.l1((1-mask) * y_true, (1-mask) * y_pred)
    
    def loss_valid(self, mask, y_true, y_pred):
        """Pixel L1 loss outside the hole / mask"""
        return self.l1(mask * y_true, mask * y_pred)
    
    def loss_perceptual(self, vgg_out, vgg_gt, vgg_comp): 
        """Perceptual loss based on VGG16, see. eq. 3 in paper"""       
        loss = 0
        for o, c, g in zip(vgg_out, vgg_comp, vgg_gt):
            loss += self.l1(o, g) + self.l1(c, g)
        return loss
        
    def loss_style(self, output, vgg_gt):
        """Style loss based on output/computation, used for both eq. 4 & 5 in paper"""
        loss = 0
        for o, g in zip(output, vgg_gt):
            loss += self.l1(self.gram_matrix(o), self.gram_matrix(g))
        return loss
    
    def loss_tv(self, mask, y_comp):
        """Total variation loss, used for smoothing the hole region, see. eq. 6"""

        # Create dilated hole region using a 3x3 kernel of all 1s.
        kernel = K.ones(shape=(3, 3, mask.shape[3], mask.shape[3]))
        dilated_mask = K.conv2d(1-mask, kernel, data_format='channels_last', padding='same')

        # Cast values to be [0., 1.], and compute dilated hole region of y_comp
        dilated_mask = K.cast(K.greater(dilated_mask, 0), 'float32')
        P = dilated_mask * y_comp

        # Calculate total variation loss
        a = self.l1(P[:,1:,:,:], P[:,:-1,:,:])
        b = self.l1(P[:,:,1:,:], P[:,:,:-1,:])        
        return a+b

    def fit(self, generator, epochs=10, plot_callback=None, *args, **kwargs):
        """Fit the U-Net to a (images, targets) generator
        
        param generator: training generator yielding (maskes_image, original_image) tuples
        param epochs: number of epochs to train for
        param plot_callback: callback function taking Unet model as parameter
        """
        
        filename = datetime.now().strftime("%Y%m%d_%H%M")
        errlog_path = cst.ERRLOG_PATH + filename + '.csv'
#         print('ErrorLog Path' + errlog_path)
        
        # Loop over epochs
#         while True: # For raise StopIteration()
        for _ in range(epochs):
            start = time.time()
            print("Start       :" + str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            
            try:            
                # Fit the model
                self.model.fit_generator(
                    generator,
                    epochs=self.current_epoch+1,
                    initial_epoch=self.current_epoch,
                    *args, **kwargs
                )
            except Exception as e:
                print(e)
                traceback_msg = traceback.format_exc()
                error_time    = datetime.now().strftime("%Y/%m/%d %H:%M:%S")

                with open(errlog_path, 'a') as f:
                    f.write('>>>' + error_time + '\n')
                    f.write(str(traceback_msg) + '\n')

            try: 
                # Update epoch 
                self.current_epoch += 1

                # After each epoch predict on test images & show them
                if plot_callback:
                    plot_callback(self.model)

                # Save logfile
                if self.weight_filepath:
                    self.save()
            except Exception as e:
                print(e)
                traceback_msg = traceback.format_exc()
                error_time    = datetime.now().strftime("%Y/%m/%d %H:%M:%S")

                with open(errlog_path, 'a') as f:
                    f.write('>>>' + error_time + '\n')
                    f.write(str(traceback_msg) + '\n')
                    
            elapsed_time = time.time() - start            
            print("Elapsed_time:{0}".format(elapsed_time) + "[sec]")   
            print("End         :" + str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

            
    def predict(self, sample):
        """Run prediction using this model"""
        return self.model.predict(sample)

    def summary(self):
        """Get summary of the UNet model"""
        print(self.model.summary())

    def save(self):        
        self.model.save_weights(self.current_weightfile())

    def load(self, filepath, train_bn=True, lr=0.0002):
        print(self.img_rows)
        print(self.img_cols)

        # Create UNet-like model
        self.model = self.build_pconv_unet(train_bn, lr)
        input_tensor = Input((self.img_rows, self.img_cols, 3))
        mask_tensor = Input((self.img_rows, self.img_cols, 3))
        out = self.model([input_tensor, mask_tensor])
        partial_model = Model([input_tensor, mask_tensor], out)

        # Load weights into model
        epoch = int(os.path.basename(filepath).split("_")[0])
        assert epoch > 0, "Could not parse weight file. Should start with 'X_', with X being the epoch"
        self.current_epoch = epoch
        partial_model.load_weights(filepath)        

    def current_weightfile(self):
        assert self.weight_filepath != None, 'Must specify location of logs'
        return self.weight_filepath + "{}_weights_{}.h5".format(self.current_epoch, self.current_timestamp())

    @staticmethod
    def current_timestamp():
        return datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    
    @staticmethod
    def l1(y_true, y_pred):
        """Calculate the L1 loss used in all loss calculations"""
        if K.ndim(y_true) == 4:
            return K.sum(K.abs(y_pred - y_true), axis=[1,2,3])
        elif K.ndim(y_true) == 3:
            return K.sum(K.abs(y_pred - y_true), axis=[1,2])
        else:
            raise NotImplementedError("Calculating L1 loss on 1D tensors? should not occur for this network")
    
    @staticmethod
    def gram_matrix(x, norm_by_channels=False):
        """Calculate gram matrix used in style loss"""
        
        # Assertions on input
        assert K.ndim(x) == 4, 'Input tensor should be a 4d (B, H, W, C) tensor'
        assert K.image_data_format() == 'channels_last', "Please use channels-last format"        
        
        # Permute channels and get resulting shape
        x = K.permute_dimensions(x, (0, 3, 1, 2))
        shape = K.shape(x)
        B, C, H, W = shape[0], shape[1], shape[2], shape[3]
        
        # Reshape x and do batch dot product
        features = K.reshape(x, K.stack([B, C, H*W]))
        gram = K.batch_dot(features, features, axes=2)
        
        # Normalize with channels, height and width
        gram = gram /  K.cast(C * H * W, x.dtype)
        
        return gram    
    
    # Add
    def load_weights(self, filepath):
        self.model.load_weights(filepath)
        
    def get_weights(self):
        import sys
        sys.setrecursionlimit(10000) # For 'RecursionError: maximum recursion depth exceeded'
        
        self.get_weights()
        
    def layers(self):
        self.layers