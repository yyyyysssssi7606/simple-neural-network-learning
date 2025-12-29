package org.nettest;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.*;

public class MnistTest {

    /**
     * 手写数字识别测试
     *
     * @param args
     */

    // 启动入口（测试用）
    public static void main(String[] args) throws IOException, ClassNotFoundException {
//        initData();

//        train();

//        test();

        SwingUtilities.invokeLater(() -> {
            try {
                UIManager.setLookAndFeel(UIManager.getLookAndFeel());
            } catch (Exception ignored) {}
            new MnistWindow().setVisible(true);
        });
    }

    private static File trainingDir = new File("./mnist-training");

    /**
     * 整理数据，训练集需要平均，对齐样本数量
     */
    public static void initData() {
        File[] files = trainingDir.listFiles();
        int minSize = 0;
        for (File file : files) {
            File[] listFiles = file.listFiles();
            if (listFiles.length < minSize || minSize == 0) {
                minSize = listFiles.length;
            }
        }

        for (File file : files) {
            File[] listFiles = file.listFiles();
            for (int i = minSize; i < listFiles.length; i++) {
                listFiles[i].delete();
            }
        }
    }


    /**
     * 设置三层神经网络
     * 逐步缩小参数
     */
    private static SimpleNetOptimizedDouble net = new SimpleNetOptimizedDouble(784, 128, 10);

    /**
     * 训练数据，然后保存为json
     * 1、创建神经网络
     * 2、遍历数据，训练神经网络，测试神经网络
     * 3、序列化数据
     * 4、加载序列化数据
     * 5、封装
     */
    public static void train() throws IOException {
        // 1、创建神经网络，并初始化
        // 2、遍历数据，训练神经网络，测试神经网络
        File[] dirs = trainingDir.listFiles();

        int size = dirs[0].listFiles().length;
        File[] files = null;
        // 进行10轮学习
        for (int i = 0; i < 10; i++) {
            for (int picIndex = 0; picIndex < size; picIndex++) {
                int num = 0;
                while (num < 10) {
                    files = dirs[num].listFiles();
                    double[] image = readImage(files[picIndex]);
                    net.training(num, image);
                    ++ num;
                }
            }
            // 打印相关信息
            net.printAllData();
            // 序列化类
            serializeObject(net, "./SimpleNetOptimized/" + net.getClass().getSimpleName() + "-" + i + ".object");
        }


        // 3、序列化数据
        // 4、加载序列化数据
        // 5、封装

    }

    /**
     * 测试
     */
    private static File testDir = new File("./mnist-testing");
    private static void test() throws IOException, ClassNotFoundException {
        // 加载类
        net = (SimpleNetOptimizedDouble) deserializeObject("./SimpleNetOptimized/" + net.getClass().getSimpleName() + "-9.object");

        File[] files = testDir.listFiles();
        int minSize = 0;
        for (File file : files) {
            File[] listFiles = file.listFiles();
            if (listFiles.length < minSize || minSize == 0) {
                minSize = listFiles.length;
            }
        }

        int correctCount = 0;
        int errorCount = 0;
        /*
         测试
         */
        for (int i = 0; i < files.length; i++) {
            File[] listFiles = files[i].listFiles();
            for (File image : listFiles) {
                double[] readImage = readImage(image);
                double[] predict = net.predict(readImage);
                double maxScore = predict[0];
                int maxIndex = 0;
                for (double score : predict) {
                    if (score > maxScore) {
                        maxScore = score;
                        maxIndex = i;
                    }
                }
                if (maxIndex == i) {
                    ++ correctCount;
                } else {
                    ++ errorCount;
                }
            }
        }

        System.out.println("正确：" + correctCount);
        System.out.println("错误：" + errorCount);
        System.out.println("正确率：" + ((double)correctCount / ((double)correctCount + (double)errorCount) * 100) + "%");
    }

    /**
     * 图片高度
     */
    private static final int PIC_IMAGE_WIDTH = 28;

    /**
     * 图片宽度
     */
    private static final int PIC_IMAGE_HEIGHT = 28;

    private static final BufferedImage grayImg = new BufferedImage(PIC_IMAGE_WIDTH, PIC_IMAGE_HEIGHT, BufferedImage.TYPE_BYTE_GRAY);
    public static double[] readImage(File file) throws IOException {
        grayImg.getGraphics().drawImage(ImageIO.read(file), 0, 0, PIC_IMAGE_WIDTH, PIC_IMAGE_HEIGHT, null);
        double[] resArr = new double[PIC_IMAGE_HEIGHT * PIC_IMAGE_WIDTH];
        for (int y = 0; y < PIC_IMAGE_HEIGHT; y++) {
            for (int x = 0; x < PIC_IMAGE_WIDTH; x++) {
                int rgb = grayImg.getRGB(x, y);
                // 转灰度：Y = 0.299R + 0.587G + 0.114B
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;
                double gray = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
                resArr[y * PIC_IMAGE_WIDTH + x] = gray;
            }
        }
        return resArr;
    }

    /**
     * 序列化对象到文件
     *
     * @param obj
     * @param filename
     * @throws IOException
     */
    public static void serializeObject(Object obj, String filename) throws IOException {
        File file = new File(filename);
        if (file.exists()) {
            file.delete();
            file.createNewFile();
        }
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filename))) {
            oos.writeObject(obj);
        }
    }

    /**
     * 从文件反序列化对象
     *
     * @param filename
     * @return
     * @throws IOException
     * @throws ClassNotFoundException
     */
    public static Object deserializeObject(String filename)
            throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filename))) {
            return ois.readObject();
        }
    }


}
