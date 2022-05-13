from tensorflow.keras.optimizers import *
import time
import matplotlib.pyplot as plt
from load_data import *
from build_model import *
from loss_function import *


class StarGAN_pretrain2_ck:
    def __init__(self):
        self.generator = build_generator()
        self.discriminator = build_discriminator()
        self.g_opt = Adam(1e-4)
        self.d_opt = Adam(1e-4)
        self.natural_train_roots, self.expression_train_roots = build_pretrain_CK_part2_data(train=True, direction='N2E')
        self.natural_test_roots, self.expression_test_roots = build_pretrain_CK_part2_data(train=False, direction='N2E')

    def gen_train_step(self, natural_source, expression_source, train=True, direction='N2E'):
        if direction == 'N2E':
            source = tf.cast(natural_source, dtype='float32')
            gt = tf.cast(expression_source, dtype='float32')
            target_label = [1] * source.shape[0]
            target_label = tf.one_hot(target_label, depth=2)
            source_label = [0] * source.shape[0]
            source_label = tf.one_hot(source_label, depth=2)
        elif direction == 'E2N':
            source = tf.cast(expression_source, dtype='float32')
            gt = tf.cast(natural_source, dtype='float32')
            target_label = [0] * source.shape[0]
            target_label = tf.one_hot(target_label, depth=2)
            source_label = [1] * source.shape[0]
            source_label = tf.one_hot(source_label, depth=2)

        with tf.GradientTape() as tape:
            gen_img = self.generator.call([source, target_label])
            v_gen, c_gen = self.discriminator.call(gen_img)
            loss_img = reconstruction_loss(gt, gen_img)
            loss_cls = classify_loss(target_label, c_gen)
            loss_adv = adversarial_loss(v_gen, True)
            loss_g = 10 * loss_img + loss_adv + 10 * loss_cls
        if train:
            grads = tape.gradient(loss_g, self.generator.trainable_variables)
            self.g_opt.apply_gradients(zip(grads, self.generator.trainable_variables))
            return loss_g
        else:
            return loss_g, 10 * loss_img, loss_adv, 10 * loss_cls

    def dis_train_step(self, natural_source, expression_source, train=True, direction='N2E'):
        if direction == 'N2E':
            source = tf.cast(natural_source, dtype='float32')
            gt = tf.cast(expression_source, dtype='float32')
            target_label = [1] * source.shape[0]
            target_label = tf.one_hot(target_label, depth=2)
            source_label = [0] * source.shape[0]
            source_label = tf.one_hot(source_label, depth=2)
        elif direction == 'E2N':
            source = tf.cast(expression_source, dtype='float32')
            gt = tf.cast(natural_source, dtype='float32')
            target_label = [0] * source.shape[0]
            target_label = tf.one_hot(target_label, depth=2)
            source_label = [1] * source.shape[0]
            source_label = tf.one_hot(source_label, depth=2)

        with tf.GradientTape() as tape:
            gen_img = self.generator.call([source, target_label])
            v_gen, c_gen = self.discriminator.call(gen_img)
            v_real_s, c_real_s = self.discriminator.call(source)
            v_real_gt, c_real_gt = self.discriminator.call(gt)
            loss_adv = 0.5 * (adversarial_loss(v_gen, False) + adversarial_loss(v_real_gt, True))
            loss_cls = classify_loss(target_label, c_real_gt) + classify_loss(source_label, c_real_s)
            loss_d = loss_adv + 10 * loss_cls
        if train:
            grads = tape.gradient(loss_d, self.discriminator.trainable_variables)
            self.d_opt.apply_gradients(zip(grads, self.discriminator.trainable_variables))
            return loss_d
        else:
            return loss_d, loss_adv, 10 * loss_cls

    def train(self, epochs=20, interval=1, batch_size=32, batch_num=104):
        tr_L_G_avg = []
        te_L_G_img_avg = []
        te_L_G_adv_avg = []
        te_L_G_cls_avg = []
        tr_L_D_avg = []
        te_L_D_adv_avg = []
        te_L_D_cls_avg = []
        te_L_G_avg = []
        te_L_D_avg = []
        start = time.time()
        for epoch in range(epochs):
            ep_start = time.time()
            tr_L_G = []
            te_L_G_img = []
            te_L_G_adv = []
            te_L_G_cls = []
            tr_L_D = []
            te_L_D_adv = []
            te_L_D_cls = []
            te_L_G = []
            te_L_D = []
            for b in range(batch_num):
                natural = load_image(get_batch_data(self.natural_train_roots, b, batch_size))
                expression = load_image(get_batch_data(self.expression_train_roots, b, batch_size))
                b_test = random.randint(0, 48)
                natural_test = load_image(get_batch_data(self.natural_test_roots, b_test, batch_size))
                expression_test = load_image(get_batch_data(self.expression_test_roots, b_test, batch_size))
                for direction in ['N2E', 'E2N']:
                    for i in range(2):
                        loss_d = self.dis_train_step(natural, expression, train=True, direction=direction)
                    loss_d_test, loss_adv_d, loss_cls_d = self.dis_train_step(natural_test, expression_test, train=False, direction=direction)
                    loss_g = self.gen_train_step(natural, expression, train=True, direction=direction)
                    loss_g_test, loss_img, loss_adv_g, loss_cls_g = self.gen_train_step(natural_test, expression_test, train=False, direction=direction)
                    tr_L_G.append(loss_g)
                    te_L_G_img.append(loss_img)
                    te_L_G_adv.append(loss_adv_g)
                    te_L_G_cls.append(loss_cls_g)
                    tr_L_D.append(loss_d)
                    te_L_D_adv.append(loss_adv_d)
                    te_L_D_cls.append(loss_cls_d)
                    te_L_G.append(loss_g_test)
                    te_L_D.append(loss_d_test)

            tr_L_G_avg.append(np.mean(tr_L_G))
            te_L_G_img_avg.append(np.mean(te_L_G_img))
            te_L_G_adv_avg.append(np.mean(te_L_G_adv))
            te_L_G_cls_avg.append(np.mean(te_L_G_cls))
            tr_L_D_avg.append(np.mean(tr_L_D))
            te_L_D_adv_avg.append(np.mean(te_L_D_adv))
            te_L_D_cls_avg.append(np.mean(te_L_D_cls))
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
            print('Test Loss Gen_img       :  {:8.5f}'.format(te_L_G_img_avg[-1]))
            print('Test Loss Gen_adv       :  {:8.5f}'.format(te_L_G_adv_avg[-1]))
            print('Test Loss Gen_cls       :  {:8.5f}'.format(te_L_G_cls_avg[-1]))
            print('Train Loss Discriminator :  {:8.5f}'.format(tr_L_D_avg[-1]))
            print('Test Loss Discriminator  :  {:8.5f}'.format(te_L_D_avg[-1]))
            print('Test Loss Dis_adv       :  {:8.5f}'.format(te_L_D_adv_avg[-1]))
            print('Test Loss Dis_cls       :  {:8.5f}'.format(te_L_D_cls_avg[-1]))

            self.sample_image(epoch)
            if (epoch % interval == 0 or epoch + 1 == epochs) and (te_L_G_avg[-1] <= np.mean(te_L_G_avg)):
                self.generator.save_weights('pretrain_weight/new_2ck_gen_weights_{}'.format(epoch + 1))
                self.discriminator.save_weights('pretrain_weight/new_2ck_dis_weights_{}'.format(epoch + 1))

        return [tr_L_G_avg, te_L_G_img_avg, te_L_G_adv_avg, te_L_G_cls_avg], \
               [tr_L_D_avg, te_L_D_adv_avg, te_L_D_cls_avg], [te_L_G_avg, te_L_D_avg]

    def sample_image(self, epoch, path='pretrain_picture/'):
        natural_train = load_image(get_batch_data(self.natural_train_roots, 0, 5))
        natural_test = load_image(get_batch_data(self.natural_test_roots, 0, 5))
        natural_sampling = tf.concat([natural_train, natural_test], axis=0)
        expression_train = load_image(get_batch_data(self.expression_train_roots, 0, 5))
        expression_test = load_image(get_batch_data(self.expression_test_roots, 0, 5))
        expression_sampling = tf.concat([expression_train, expression_test], axis=0)
        expression_label = [1] * expression_sampling.shape[0]
        expression_label = tf.one_hot(expression_label, depth=2)
        natural_label = [0] * natural_sampling.shape[0]
        natural_label = tf.one_hot(natural_label, depth=2)

        gen_natural = self.generator.predict([expression_sampling, natural_label])
        gen_expression = self.generator.predict([natural_sampling, expression_label])
        cycle_natural = self.generator.predict([gen_expression, natural_label])
        cycle_expression = self.generator.predict([gen_natural, expression_label])

        natural_sampling = 0.5 * (natural_sampling + 1)
        expression_sampling = 0.5 * (expression_sampling + 1)
        gen_natural = 0.5 * (gen_natural + 1)
        gen_expression = 0.5 * (gen_expression + 1)
        cycle_natural = 0.5 * (cycle_natural + 1)
        cycle_expression = 0.5 * (cycle_expression + 1)

        r, c = 6, 10
        fig, axs = plt.subplots(r, c, sharex='col', sharey='row', figsize=(25, 25))
        plt.subplots_adjust(hspace=0.2)
        cnt = 0
        for j in range(c):
            axs[0, j].imshow(natural_sampling[cnt], cmap='gray')
            axs[0, j].axis('off')
            axs[1, j].imshow(gen_expression[cnt], cmap='gray')
            axs[1, j].axis('off')
            axs[2, j].imshow(cycle_natural[cnt], cmap='gray')
            axs[2, j].axis('off')
            axs[3, j].imshow(expression_sampling[cnt], cmap='gray')
            axs[3, j].axis('off')
            axs[4, j].imshow(gen_natural[cnt], cmap='gray')
            axs[4, j].axis('off')
            axs[5, j].imshow(cycle_expression[cnt], cmap='gray')
            axs[5, j].axis('off')
            cnt += 1
        fig.savefig(path + 'new_2_{}.png'.format(epoch + 1))
        plt.close()

if __name__ == '__main__':
    stargan = StarGAN_pretrain2_ck()
    stargan.generator.load_weights('pretrain_weight/new_ck_gen_weights_3')
    stargan.discriminator.load_weights('pretrain_weight/new_ck_dis_weights_3')
    [tr_L_G_avg, te_L_G_img_avg, te_L_G_adv_avg, te_L_G_cls_avg], [tr_L_D_avg, te_L_D_adv_avg, te_L_D_cls_avg], \
    [te_L_G_avg, te_L_D_avg] = stargan.train(epochs=10, interval=1)

    plt.plot(tr_L_G_avg)
    plt.title('Generator total loss')
    plt.savefig('pretrain_picture/new_2_Generator loss.jpg')
    plt.close()

    plt.plot(te_L_G_adv_avg)
    plt.plot(te_L_D_adv_avg)
    plt.legend(['Generator', 'Discriminator'])
    plt.title('Adversarial pretrain loss')
    plt.savefig('pretrain_picture/new_2_Adversarial loss.jpg')
    plt.close()

    plt.plot(te_L_G_img_avg)
    plt.title('Generator Image loss')
    plt.savefig('pretrain_picture/new_2_Generator Image loss.jpg')
    plt.close()

    plt.plot(te_L_G_cls_avg)
    plt.title('Generator Classify loss')
    plt.savefig('pretrain_picture/new_2_Generator Classify loss.jpg')
    plt.close()

    plt.plot(tr_L_D_avg)
    plt.title('Discriminator total loss')
    plt.savefig('pretrain_picture/new_2_Discriminator loss.jpg')
    plt.close()

    plt.plot(te_L_D_cls_avg)
    plt.title('Discriminator Classify loss')
    plt.savefig('pretrain_picture/new_2_Discriminator Classify loss.jpg')
    plt.close()

    plt.plot(te_L_G_avg)
    plt.title('Generator test loss')
    plt.savefig('pretrain_picture/new_2_Generator test loss.jpg')
    plt.close()

    plt.plot(te_L_D_avg)
    plt.title('Discriminator test loss')
    plt.savefig('picture/new_2_Discriminator test loss.jpg')
    plt.close()
