from tensorflow.keras.optimizers import *
import time
import matplotlib.pyplot as plt
from load_data import *
from build_model import *
from loss_function import *


class StarGAN_pretrain:
    def __init__(self):
        self.generator = build_generator()
        self.discriminator = build_discriminator()
        self.g_opt = Adam(1e-4)
        self.d_opt = Adam(1e-4)
        self.train_dataset, self.test_dataset = load_celebA()

    def gen_train_step(self, source, train=True):
        source = tf.cast(source, dtype='float32')
        label = tf.zeros((source.shape[0], 2))
        with tf.GradientTape() as tape:
            pred_img = self.generator.call([source, label])
            v_gen, c_gen = self.discriminator.call(pred_img)
            loss_img = reconstruction_loss(source, pred_img)
            loss_adv = adversarial_loss(v_gen, True)
            loss_g = loss_img + loss_adv
        if train:
            grads = tape.gradient(loss_g, self.generator.trainable_variables)
            self.g_opt.apply_gradients(zip(grads, self.generator.trainable_variables))
            return loss_g
        else:
            return loss_g, loss_img, loss_adv

    def dis_train_step(self, source, train=True):
        source = tf.cast(source, dtype='float32')
        label = tf.zeros((source.shape[0], 2))
        with tf.GradientTape() as tape:
            pred_img = self.generator.call([source, label])
            v_gen, c_gen = self.discriminator.call(pred_img)
            v_real, c_real = self.discriminator.call(source)
            loss_adv = 0.5 * (adversarial_loss(v_gen, False) + adversarial_loss(v_real, True))
            loss_d = loss_adv
        if train:
            grads = tape.gradient(loss_d, self.discriminator.trainable_variables)
            self.d_opt.apply_gradients(zip(grads, self.discriminator.trainable_variables))
            return loss_d
        else:
            return loss_d, loss_adv

    def train(self, epochs=20, interval=1, batch_size=32, batch_num=1000):
        tr_L_G_avg = []
        te_L_G_img_avg = []
        te_L_G_adv_avg = []
        tr_L_D_avg = []
        te_L_D_adv_avg = []
        te_L_G_avg = []
        te_L_D_avg = []
        start = time.time()
        for epoch in range(epochs):
            ep_start = time.time()
            tr_L_G = []
            te_L_G_img = []
            te_L_G_adv = []
            tr_L_D = []
            te_L_D_adv = []
            te_L_G = []
            te_L_D = []
            for b in range(batch_num):
                source = load_image(get_batch_data(self.train_dataset, b, batch_size))
                b_test = random.randint(0, 99)
                source_test = load_image(get_batch_data(self.test_dataset, b_test, batch_size))
                loss_d = self.dis_train_step(source)
                loss_d_test, loss_adv_d = self.dis_train_step(source_test, train=False)
                loss_g = self.gen_train_step(source)
                loss_g_test, loss_img, loss_adv_g = self.gen_train_step(source_test, train=False)
                tr_L_G.append(loss_g)
                te_L_G_img.append(loss_img)
                te_L_G_adv.append(loss_adv_g)
                tr_L_D.append(loss_d)
                te_L_D_adv.append(loss_adv_d)
                te_L_G.append(loss_g_test)
                te_L_D.append(loss_d_test)

            tr_L_G_avg.append(np.mean(tr_L_G))
            te_L_G_img_avg.append(np.mean(te_L_G_img))
            te_L_G_adv_avg.append(np.mean(te_L_G_adv))
            tr_L_D_avg.append(np.mean(tr_L_D))
            te_L_D_adv_avg.append(np.mean(te_L_D_adv))
            te_L_G_avg.append(np.mean(te_L_G))
            te_L_D_avg.append(np.mean(te_L_D))

            t_pass = time.time() - start
            m_pass, s_pass = divmod(t_pass, 60)
            h_pass, m_pass = divmod(m_pass, 60)
            print('\nTime for pass  {:<4d}: {:<2d} hour {:<3d} min {:<4.3f} sec'.format(epoch + 1, int(h_pass),
                                                                                        int(m_pass), s_pass))
            print('Time for epoch {:<4d}: {:6.3f} sec'.format(epoch + 1, time.time() - ep_start))
            print('Train Loss Generator     :  {:8.5f}'.format(tr_L_G_avg[-1]))
            print('Test Loss Generator      :  {:8.5f}'.format(te_L_G_avg[-1]))
            print('Test Loss Gen_adv       :  {:8.5f}'.format(te_L_G_adv_avg[-1]))
            print('Test Loss Gen_img       :  {:8.5f}'.format(te_L_G_img_avg[-1]))
            print('Train Loss Discriminator :  {:8.5f}'.format(tr_L_D_avg[-1]))
            print('Test Loss Discriminator  :  {:8.5f}'.format(te_L_D_avg[-1]))
            print('Test Loss Dis_adv       :  {:8.5f}'.format(te_L_D_adv_avg[-1]))

            self.sample_image_pretrain(epoch)
            if (epoch % interval == 0 or epoch + 1 == epochs) and (te_L_G_avg[-1] <= np.min(te_L_G_avg)):
                self.generator.save_weights('pretrain_weight/new_celeb_gen_weights_{}'.format(epoch + 1))
                self.discriminator.save_weights('pretrain_weight/new_celeb_dis_weights_{}'.format(epoch + 1))

        return [tr_L_G_avg, te_L_G_adv_avg, te_L_G_img_avg], [tr_L_D_avg, te_L_D_adv_avg], \
               [te_L_G_avg, te_L_D_avg]

    def sample_image_pretrain(self, epoch, path='pretrain_picture/new_pretrain'):
        source_train = load_image(get_batch_data(self.train_dataset, 0, 5))
        source_test = load_image(get_batch_data(self.test_dataset, 0, 5))
        source_sampling = tf.concat([source_train, source_test], axis=0)
        label_sampling = tf.zeros((source_sampling.shape[0], 2))

        gen_img = self.generator.predict([source_sampling, label_sampling])

        source_sampling = 0.5 * (source_sampling + 1)
        gen_img = 0.5 * (gen_img + 1)

        r, c = 2, 10
        fig, axs = plt.subplots(r, c, sharex='col', sharey='row', figsize=(25, 25))
        plt.subplots_adjust(hspace=0.2)
        cnt = 0
        for j in range(c):
            axs[0, j].imshow(source_sampling[cnt], cmap='gray')
            axs[0, j].axis('off')
            axs[1, j].imshow(gen_img[cnt], cmap='gray')
            axs[1, j].axis('off')
            cnt += 1
        fig.savefig(path + '_{}.png'.format(epoch + 1))
        plt.close()


if __name__ == '__main__':
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession
    import os

    print(tf.__version__)
    print(tf.test.is_gpu_available())
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    config = ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    stargan = StarGAN_pretrain()

    [tr_L_G_avg, te_L_G_adv_avg, te_L_G_img_avg], [tr_L_D_avg, te_L_D_adv_avg], \
    [te_L_G_avg, te_L_D_avg] = stargan.train(epochs=5, interval=1)

    plt.plot(tr_L_G_avg)
    plt.title('Generator pretrain loss')
    plt.savefig('pretrain_picture/new_Generator loss.jpg')
    plt.close()

    plt.plot(te_L_G_adv_avg)
    plt.plot(te_L_D_adv_avg)
    plt.legend(['Generator', 'Discriminator'])
    plt.title('Adversarial pretrain loss')
    plt.savefig('pretrain_picture/new_Adversarial loss.jpg')
    plt.close()

    plt.plot(te_L_G_img_avg)
    plt.title('Generator pretrain Image loss')
    plt.savefig('pretrain_picture/new_Generator Image loss.jpg')
    plt.close()

    plt.plot(tr_L_D_avg)
    plt.title('Discriminator pretrain loss')
    plt.savefig('pretrain_picture/new_Discriminator pretrain loss.jpg')
    plt.close()

    plt.plot(te_L_G_avg)
    plt.title('Generator test loss')
    plt.savefig('pretrain_picture/new_Generator test loss.jpg')
    plt.close()

    plt.plot(te_L_D_avg)
    plt.title('Discriminator pretrain test loss')
    plt.savefig('pretrain_picture/new_Discriminator test loss.jpg')
    plt.close()


