import matplotlib.pyplot as plt
import cv2
from build_model import *
from load_data import *

stargan_generator = build_generator()
stargan_generator.load_weights('weight/gen_weights_13')
condition = 6

if condition == 0:
    train = True
    if train: total_id = 46
    else : total_id = 33
    for i in range(total_id):
        source, _, id_ = load_ck_by_id(i, train)
        expression_label = [1] * source.shape[0]
        expression_label = tf.one_hot(expression_label, depth=2)
        natural_label = [0] * source.shape[0]
        natural_label = tf.one_hot(natural_label, depth=2)

        gen_img = stargan_generator.predict([source, expression_label])
        gen_img2 = stargan_generator.predict(([gen_img, natural_label]))

        source = 0.5 * (source + 1)
        gen_img = 0.5 * (gen_img + 1)
        gen_img2 = 0.5 * (gen_img2 + 1)

        r, c = 3, source.shape[0]
        fig, axs = plt.subplots(r, c, sharex='col', sharey='row', figsize=(25, 25))
        plt.subplots_adjust(hspace=0.2)
        cnt = 0
        for j in range(c):
            axs[0, j].imshow(source[cnt], cmap='gray')
            axs[0, j].axis('off')
            axs[1, j].imshow(gen_img[cnt], cmap='gray')
            axs[1, j].axis('off')
            axs[2, j].imshow(gen_img2[cnt], cmap='gray')
            axs[2, j].axis('off')
            cnt += 1
        fig.savefig('picture/cond0/ep96/{}'.format(id_))
        plt.close()

elif condition == 1:
    train = False
    if train:
        total_id = 46
    else:
        total_id = 33
    for i in range(total_id):
        source, img_path, id_ = load_natural_ck_by_id(i, train)
        expression_label = [1] * source.shape[0]
        expression_label = tf.one_hot(expression_label, depth=2)

        gen_img = stargan_generator.predict([source, expression_label])

        gen_img = (255 * 0.5 * (gen_img + 1)).astype('uint8')
        file_path = 'picture/cond1/ep96/{}'.format(id_)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        for i in range(gen_img.shape[0]):
            img_name = img_path[i].split('/')[-1]
            cv2.imwrite(file_path + '/' + img_name, gen_img[i])

elif condition == 2:
    for train in [True, False]:
        if train:
            total_id = 46
        else:
            total_id = 33
        for i in range(total_id):
            source, img_path, id_ = load_ck_by_id(i, train, emo='expression')
            natural_label = [0] * source.shape[0]
            natural_label = tf.one_hot(natural_label, depth=2)

            gen_img = stargan_generator.predict([source, natural_label])

            gen_img = (255 * 0.5 * (gen_img + 1)).astype('uint8')
            file_path = 'picture/cond2/1img_loss/{}'.format(id_)
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            for i in range(gen_img.shape[0]):
                img_name = img_path[i].split('/')[-1]
                cv2.imwrite(file_path + '/' + img_name, gen_img[i])

elif condition == 3:
    path = '/home/pomelo96/Desktop/datasets/harry'
    img_path = os.listdir(path)

    for name in img_path:
        source = cv2.imread(path + '/' + name)
        source = cv2.resize(source, (128, 128))
        source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        source = np.expand_dims(source, axis=-1)
        source = tf.reshape(source, (1, source.shape[0], source.shape[1], source.shape[2]))
        source = (source / 255) * 2 - 1

        label = [1]*source.shape[0]
        label = tf.one_hot(label, depth=2)

        gen_img = stargan_generator.predict([source, label])
        for i in range(gen_img.shape[0]):
            img = gen_img[i]
            img = (255 * (0.5 * (img + 1))).astype('uint8')
            save_path = 'picture/cond3/harry_StarGAN/' + name
            cv2.imwrite(save_path, img)

elif condition == 4:
    path = '/home/pomelo96/Desktop/datasets/CMU/'
    emo_type = os.listdir(path)
    emo_type.sort()
    for emo in emo_type:
        emo_path = path + emo
        for t in ['train', 'test']:
            t_path = emo_path + '/' + t
            img_name_list = os.listdir(t_path)
            img_name_list.sort()
            img_path_list = []
            [img_path_list.append(t_path + '/' + img_name) for img_name in img_name_list]

            source = load_image(img_path_list)
            if emo == 'e':
                label = [0] * source.shape[0]
            elif emo == 'n':
                label = [1] * source.shape[0]
            label = tf.one_hot(label, depth=2)
            gen_img = stargan_generator.predict([source, label])

            for i in range(gen_img.shape[0]):
                img = gen_img[i]
                img = (255 * (0.5 * (img + 1))).astype('uint8')
                if emo == 'e':
                    save_file = 'picture/cond4/E2N/' + t
                elif emo == 'n':
                    save_file = 'picture/cond4/N2E/' + t
                save_path = save_file + '/' + img_name_list[i]
                cv2.imwrite(save_path, img)

elif condition == 5:
    path = '/home/pomelo96/Desktop/datasets/Natural image'
    id_list = os.listdir(path)
    id_list.sort()
    for id_ in id_list:
        id_path = path + '/' + id_
        id_path = id_path + '/' + os.listdir(id_path)[0]
        img_name_list = os.listdir(id_path)
        img_name_list.sort()

        img_path_list = []
        [img_path_list.append(id_path + '/' + img_name) for img_name in img_name_list]

        source = load_image((img_path_list))
        label = [1] * source.shape[0]
        label = tf.one_hot(label, depth=2)
        gen_img = stargan_generator.predict([source, label])

        for i in range(gen_img.shape[0]):
            img = gen_img[i]
            img = (255 * (0.5 * (img + 1))).astype('uint8')

            save_file = 'picture/cond5/' + id_
            if not os.path.exists(save_file):
                os.makedirs(save_file)
            save_path = save_file + '/' + img_name_list[i]
            cv2.imwrite(save_path, img)

elif condition == 6:
    path = '/home/pomelo96/Desktop/datasets/Expression image'
    id_list = os.listdir(path)
    id_list.sort()
    for id_ in id_list:
        id_path = path + '/' + id_
        id_path = id_path + '/' + os.listdir(id_path)[0]
        img_name_list = os.listdir(id_path)
        img_name_list.sort()

        img_path_list = []
        [img_path_list.append(id_path + '/' + img_name) for img_name in img_name_list]

        source = load_image((img_path_list))
        label = [0] * source.shape[0]
        label = tf.one_hot(label, depth=2)
        gen_img = stargan_generator.predict([source, label])

        for i in range(gen_img.shape[0]):
            img = gen_img[i]
            img = (255 * (0.5 * (img + 1))).astype('uint8')

            save_file = 'picture/cond6/' + id_
            if not os.path.exists(save_file):
                os.makedirs(save_file)
            save_path = save_file + '/' + img_name_list[i]
            cv2.imwrite(save_path, img)
