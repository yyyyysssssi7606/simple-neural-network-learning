package org.nettest;

import javax.swing.*;

public class MnistTest {

    /**
     * 手写数字识别测试
     *
     * @param args
     */

    // 启动入口（测试用）
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            try {
                UIManager.setLookAndFeel(UIManager.getLookAndFeel());
            } catch (Exception ignored) {}
            new MnistWindow().setVisible(true);
        });
    }


}
