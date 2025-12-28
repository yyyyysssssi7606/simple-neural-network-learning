package org.nettest;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;

public class MnistWindow extends JFrame {

    /**
     * 创建手写数字识别的神经网络
     */
    private static final String MODAL_PATH = "./SimpleNetOptimized/SimpleNetOptimized-9.object";
    private static final SimpleNetOptimized modal = (SimpleNetOptimized) deserializeObject(MODAL_PATH);

    private static final int LOGICAL_WIDTH = 28;
    private static final int LOGICAL_HEIGHT = 28;
    // 缩放倍数：28*10 = 280px 显示画布
    private static final int SCALE = 10;
    private static final int CANVAS_WIDTH = LOGICAL_WIDTH * SCALE;
    private static final int CANVAS_HEIGHT = LOGICAL_HEIGHT * SCALE;

    // true=黑
    private final int[] pixels = new int[LOGICAL_HEIGHT * LOGICAL_WIDTH];
    private final JPanel canvasPanel = new JPanel() {
        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            Graphics2D g2d = (Graphics2D) g;
            g2d.setColor(Color.WHITE);
            g2d.fillRect(0, 0, getWidth(), getHeight());
            g2d.setColor(Color.BLACK);

            // 绘制网格（可选，方便对齐）
            // g2d.setStroke(new BasicStroke(0.5f));
            // for (int i = 0; i <= LOGICAL_WIDTH; i++) g2d.drawLine(i * SCALE, 0, i * SCALE, CANVAS_HEIGHT);
            // for (int i = 0; i <= LOGICAL_HEIGHT; i++) g2d.drawLine(0, i * SCALE, CANVAS_WIDTH, i * SCALE);

            // 绘制已激活像素
            for (int y = 0; y < LOGICAL_HEIGHT; y++) {
                for (int x = 0; x < LOGICAL_WIDTH; x++) {
                    if (pixels[y * LOGICAL_WIDTH + x] == 1) {
                        g2d.fillRect(x * SCALE, y * SCALE, SCALE, SCALE);
                    }
                }
            }
        }
    };

    private JLabel resultLabel = new JLabel("识别结果：暂无");

    public MnistWindow() {
        setTitle("手写数字识别测试");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setResizable(false);  // 不可调整大小

        // 设置主布局：左右结构
        setLayout(new BorderLayout());

        // 画布区域
        canvasPanel.setPreferredSize(new Dimension(CANVAS_WIDTH, CANVAS_HEIGHT));
        canvasPanel.setBackground(Color.WHITE);
        canvasPanel.setBorder(BorderFactory.createTitledBorder("请在此书写数字（0-9）"));
        setupDrawing();

        // 右侧控制区
        JPanel controlPanel = new JPanel();
        controlPanel.setLayout(new BoxLayout(controlPanel, BoxLayout.Y_AXIS));
        controlPanel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        JButton clearBtn = new JButton("清空");
        clearBtn.addActionListener(e -> clearCanvas());

        JButton recognizeBtn = new JButton("识别");
        recognizeBtn.addActionListener(e -> recognize());



        controlPanel.add(clearBtn);
        controlPanel.add(Box.createVerticalStrut(10));
        controlPanel.add(recognizeBtn);
        controlPanel.add(Box.createVerticalStrut(10));
        controlPanel.add(resultLabel);

        // 组合
        add(canvasPanel, BorderLayout.CENTER);
        add(controlPanel, BorderLayout.EAST);

        pack(); // 自适应内容大小
        setLocationRelativeTo(null); // 居中
    }

    private void setupDrawing() {
        canvasPanel.addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                drawAt(e.getX(), e.getY());
            }
        });
        canvasPanel.addMouseMotionListener(new MouseAdapter() {
            @Override
            public void mouseDragged(MouseEvent e) {
                drawAt(e.getX(), e.getY());
            }
        });
    }

    /**
     * 1像素画笔
     */
    private void drawAt(int x, int y) {
        // 转为逻辑坐标（0~27）
        int gridX = Math.max(0, Math.min(LOGICAL_WIDTH - 1, x / SCALE));
        int gridY = Math.max(0, Math.min(LOGICAL_HEIGHT - 1, y / SCALE));
        pixels[gridY * LOGICAL_WIDTH + gridX] = 1;
        canvasPanel.repaint();
    }

    /**
     * 2像素画笔
     * @param x
     * @param y
     */
//    private void drawAt(int x, int y) {
//        int gridX = Math.max(0, Math.min(LOGICAL_WIDTH - 1, x / SCALE));
//        int gridY = Math.max(0, Math.min(LOGICAL_HEIGHT - 1, y / SCALE));
//
//        // 🖌️ 画笔半径 = 1 → 覆盖 (gridX±1, gridY±1) 共 3×3 区域
//        for (int dy = -1; dy <= 1; dy++) {
//            for (int dx = -1; dx <= 1; dx++) {
//                int nx = gridX + dx;
//                int ny = gridY + dy;
//                if (nx >= 0 && nx < LOGICAL_WIDTH && ny >= 0 && ny < LOGICAL_HEIGHT) {
//                    pixels[ny * LOGICAL_WIDTH + nx] = 1;
//                }
//            }
//        }
//        canvasPanel.repaint();
//    }

    private void clearCanvas() {
        for (int y = 0; y < LOGICAL_HEIGHT; y++) {
            for (int x = 0; x < LOGICAL_WIDTH; x++) {
                pixels[y * LOGICAL_WIDTH + x] = 0;
            }
        }
        canvasPanel.repaint();
    }

    // 后续可在此接入你的推理逻辑：将 pixels → 归一化 float[784] → 模型预测
    private void recognize() {
        // 示例：输出当前像素分布摘要（调试用）

        double[] predict = modal.predict(pixels);
        double maxScore = predict[0];
        int maxIndex = 0;
        for (int i = 0; i < predict.length; i++) {
            double score = predict[i];
            if (score > maxScore) {
                maxScore = score;
                maxIndex = i + 1;
            }
        }

        JOptionPane.showMessageDialog(this,
                "当前识别结果：" + maxIndex,
                "识别结果", JOptionPane.INFORMATION_MESSAGE);
    }

    /**
     * 从文件反序列化对象
     *
     * @param filename
     * @return
     * @throws IOException
     * @throws ClassNotFoundException
     */
    public static Object deserializeObject(String filename) {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filename))) {
            return ois.readObject();
        } catch (IOException | ClassNotFoundException e) {
            throw new RuntimeException(e);
        }
    }

}