package org.nettest;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;

/**
 * 简单三层神经网络的优化版本
 */
public class SimpleNetOptimizedDouble implements Serializable {

    private static final long serialVersionUID = 197420123977911L;

    /**
     * 固定输入层、隐藏层、输出层大小（核心数据）
     */
    private int inputLayerSize = 9;
    private int hiddenLayerSize = 5;
    private int outputLayerSize = 4;

    /**
     * 学习率
     */
    private double learningRate = 0.1;

    /**
     *
     */
    private double lambda = 0.0001;

    /**
     * 输入层（临时变量，不参与持久化）
     */
    private double[] inputLayer;

    /**
     * 训练时使用的答案（临时变量，不参与持久化）
     */
    private int answer;

    /**
     * 权重层，数量是输入层数量 * 计算结果层数量（核心数据）
     */
    private double[] inputAndHiddenWeightLayer;

    /**
     * 权重梯度
     */
//    private double[] inputAndHiddenGradLayer;

    /**
     * 偏导数，用于平移函数（核心数据）
     */
    private double[] hiddenLayerBiasArray;

    /**
     * 输入层和权重层的计算结果层，一般输出层的结果数量要比输入层更少（临时变量，用于存储计算结果，不参与持久化）
     */
    private double[] hideLayer;

    /**
     * 权重层，数量是输入层数量 * 输入结果层数量（核心数据）
     */
    private double[] hideAndOutputWeightLayer;

    /**
     * 权重梯度
     */
//    private double[] hideAndOutputGradLayer;

    /**
     * 偏导数，用于平移函数（核心数据）
     */
    private double[] outputLayerBiasArray;

    /**
     * 输出层，一般输出层的结果数量要比输入层更少（临时变量，用于存储计算结果，不参与持久化）
     */
    private double[] outputLayer;


    /**
     * 8、输入预测数据
     *
     * @param dataArray
     */
    public void setInput(double...dataArray) {
        if (dataArray.length != inputLayerSize) {
            throw new RuntimeException("输入数据量必须为" + inputLayerSize + "个，当前数组大小为：" + dataArray.length);
        }
        for (int i = 0; i < dataArray.length; i++) {
            this.inputLayer[i] = dataArray[i];
        }
    }

    /**
     * 设置答案
     */
    public void setAnswer(int answer) {
        this.answer = answer;
    }

    private Random random = new Random(System.currentTimeMillis());


    /**
     * 创建神经网络
     *
     * @param inputLayerSize
     * @param hiddenLayerSize
     * @param outputLayerSize
     */
    public SimpleNetOptimizedDouble(int inputLayerSize, int hiddenLayerSize, int outputLayerSize) {
        if(inputLayerSize < 3) {
            throw new RuntimeException("输入层大小不能小于3");
        }
        this.inputLayerSize = inputLayerSize;
        if(hiddenLayerSize < 3) {
            throw new RuntimeException("隐藏层大小不能小于3");
        }
        this.hiddenLayerSize = hiddenLayerSize;
        if(outputLayerSize < 3) {
            throw new RuntimeException("输出层大小不能小于3");
        }
        this.outputLayerSize = outputLayerSize;
        initLayer();
    }

    /**
     * 0、初始化数据
     */
    private void initLayer() {

        inputLayer = new double[inputLayerSize];
        inputAndHiddenWeightLayer = new double[inputLayerSize * hiddenLayerSize];
//        inputAndHiddenGradLayer = new double[inputLayerSize * hiddenLayerSize];
        hiddenLayerBiasArray = new double[hiddenLayerSize];
        hideLayer = new double[hiddenLayerSize];
        hideAndOutputWeightLayer = new double[hiddenLayerSize * outputLayerSize];
//        hideAndOutputGradLayer = new double[hiddenLayerSize * outputLayerSize];
        outputLayerBiasArray = new double[outputLayerSize];
        outputLayer = new double[outputLayerSize];

        randomValue(inputAndHiddenWeightLayer);
        randomValue(hideAndOutputWeightLayer);
        randomValue(hiddenLayerBiasArray);
        randomValue(outputLayerBiasArray);
    }

    /**
     * 随机化数组值
     * @param arr
     */
    private void randomValue(double[] arr) {
        for (int i = 0; i < arr.length; i++) {
            // 初始化为-1到1范围内的随机数
            arr[i] = 2.0 * random.nextDouble() - 1.0;
        }
    }

    /**
     * 1、输入训练数据
     */
    public void addTrainData(int answer, double... dataArray) {
        setInput(dataArray);
        if (answer >= outputLayerSize || answer < 0) {
            throw new RuntimeException("输入的答案必须为0-" + outputLayerSize + "的范围！");
        }
    }

    /**
     * 2、训练
     */

    public void training(int answer, double...input) {
        setAnswer(answer);
        setInput(input);
        training();
    }

    /**
     * 训练方法
     */
    private void training() {

        double[] softmax = predict(inputLayer);

        double[] answerVector = answerConstructionVector();

        // 求o的导数，y是正确答案的onehot编码
        // dL/do[k] = softmax[k] - y[k]
        double[] outputError = new double[outputLayerSize];
        for (int outputPoint = 0; outputPoint < outputLayerSize; outputPoint++) {
            outputError[outputPoint] = softmax[outputPoint] - answerVector[outputPoint];
        }

        int hideAndOutputWeightLayerPoint = 0;
        for (int outputPoint = 0; outputPoint < outputLayerSize; outputPoint++) {
            for (int hiddenPoint = 0; hiddenPoint < hiddenLayerSize; hiddenPoint++) {
                // 这里计算要更新的层位置，比如0-100，每层10个房间，第0层的十位数就是0，加上个位数inputPoint就是对应房间号码
                hideAndOutputWeightLayerPoint = (outputPoint * hiddenLayerSize) + hiddenPoint;
                hideAndOutputWeightLayer[hideAndOutputWeightLayerPoint] += -learningRate * outputError[outputPoint] * hideLayer[hiddenPoint];
            }
            // 更新偏置层
            // ∂L/∂bⱼ = ∂L/∂oⱼ × 1
            outputLayerBiasArray[outputPoint] += -learningRate * outputError[outputPoint];
        }

        // 计算隐藏层误差
        double[] hiddenError = new double[hiddenLayerSize];
        for (int hiddenPoint = 0; hiddenPoint < hiddenLayerSize; hiddenPoint++) {
            double sum = 0;
            for (int outputPoint = 0; outputPoint < outputLayerSize; outputPoint++) {
                // 这里计算要更新的层位置，比如0-100，每层10个房间，第0层的十位数就是0，加上个位数inputPoint就是对应房间号码
                hideAndOutputWeightLayerPoint = (outputPoint * hiddenLayerSize) + hiddenPoint;
                sum += outputError[outputPoint] * hideAndOutputWeightLayer[hideAndOutputWeightLayerPoint];
            }
            hiddenError[hiddenPoint] = sum * hideLayer[hiddenPoint] * (1 - hideLayer[hiddenPoint]);
        }

        double inputNum = 0;
        int inputAndHideWeightLayerPoint = 0;
        // 更新隐藏层权重，以及隐藏层的偏置参数
        for (int hiddenPoint = 0; hiddenPoint < hiddenLayerSize; hiddenPoint++) {
            for (int inputPoint = 0; inputPoint < inputLayerSize; inputPoint++) {
                inputNum = inputLayer[inputPoint];
                // 这里计算要更新的层位置，比如0-100，每层10个房间，第0层的十位数就是0，加上个位数inputPoint就是对应房间号码
                inputAndHideWeightLayerPoint = (hiddenPoint * inputLayerSize) + inputPoint;
                inputAndHiddenWeightLayer[inputAndHideWeightLayerPoint] += -learningRate * hiddenError[hiddenPoint] * inputNum;
            }
            hiddenLayerBiasArray[hiddenPoint] += -learningRate * hiddenError[hiddenPoint];
        }

    }

    /**
     * 3、预测
     * @param input
     */
    public double[] predict(double...input) {
        setInput(input);
        //=============== 前向传播 =================
        // 等待更新的隐藏层位置游标
        int inputAndHideWeightLayerPoint = 0;
        // 当前输入层
        double inputNum = 0.0;
        // 暂存计算结果
        double calc = 0.0;
        for (int hiddenPoint = 0; hiddenPoint < hiddenLayerSize; hiddenPoint++) {
            for (int inputPoint = 0; inputPoint < inputLayerSize; inputPoint++) {
                inputNum = inputLayer[inputPoint];
                // 这里计算要更新的层位置，比如0-100，每层10个房间，第0层的十位数就是0，加上个位数inputPoint就是对应房间号码
                inputAndHideWeightLayerPoint = (hiddenPoint * inputLayerSize) + inputPoint;
                // 计算权重
                calc = calc + (inputNum * inputAndHiddenWeightLayer[inputAndHideWeightLayerPoint]);
            }
            // 添加偏导数（用于平移函数图像）
            calc = calc + hiddenLayerBiasArray[hiddenPoint];
            // 计算完成，更新到hiddenLayer中
            hideLayer[hiddenPoint] = sigmoid(calc);
            // 重置计算缓存
            calc = 0.0;
        }

        // 等待更新的隐藏层位置游标
        int hideAndOutputWeightLayerPoint = 0;
        // 当前隐藏层
        double hiddenNum = 0.0;
        // 暂存计算结果
        calc = 0.0;
        for (int outputPoint = 0; outputPoint < outputLayerSize; outputPoint++) {
            for (int hiddenPoint = 0; hiddenPoint < hiddenLayerSize; hiddenPoint++) {
                hiddenNum = hideLayer[hiddenPoint];
                // 这里计算要更新的层位置，比如0-100，每层10个房间，第0层的十位数就是0，加上个位数inputPoint就是对应房间号码
                hideAndOutputWeightLayerPoint = (outputPoint * hiddenLayerSize) + hiddenPoint;
                calc = calc + (hiddenNum * hideAndOutputWeightLayer[hideAndOutputWeightLayerPoint]);
            }
            // 添加偏导数（用于平移函数图像）
            calc = calc + outputLayerBiasArray[outputPoint];
            // 计算完成，更新到outputLayer中
            outputLayer[outputPoint] = calc;
            // 重置计算缓存
            calc = 0.0;
        }

        double[] softmax = softmax();

        return softmax;
    }

    /**
     * 激活函数，
     * @param inputNumber
     * @return
     */
    private double sigmoid(double inputNumber) {
        if (inputNumber >= 0) {
            double t = Math.exp(-inputNumber);
            return 1 / (1 + t);
        } else {
            double t = Math.exp(inputNumber);
            return t / (1 + t);
        }
    }

    /**
     * 根据当前神经网络，计算出最终的结果，用于显示
     */
    private double[] softmax() {
        double[] softmaxArray = new double[outputLayerSize];
        // 找最大值（防溢出）
        double max = outputLayer[0];
        for (int i = 1; i < outputLayerSize; i++) {
            if (outputLayer[i] > max) {
                max = outputLayer[i];
            }
        }
        // exp(z_i - max)
        double sum = 0.0;
        for (int i = 0; i < outputLayerSize; i++) {
            softmaxArray[i] = Math.exp(outputLayer[i] - max);
            sum += softmaxArray[i];
        }
        // 归一化
        for (int i = 0; i < outputLayerSize; i++) {
            softmaxArray[i] /= sum;
        }
        return softmaxArray;
    }

    /**
     * 正则化
     *
     * @return
     */
    private double l2Regularization(double[] weightList) {
        double sum = 0.0;
        // 遍历权重矩阵 W，累加 w^2
        for (double w : weightList) {
            // w²
            sum += w * w;
        }
        // λ·Σw²
        return lambda * sum;
    }

    /**
     * 计算梯度时：对 W 的每个元素加 2λw 项
     *
     * @param weightList
     * @param gradWeightList
     */
    public void addL2Gradient(double[] weightList, double[] gradWeightList) {
        for (int weightPoint = 0; weightPoint < weightList.length; weightPoint++) {
            // grad_W[i][j] 已由反向传播算出，此处原地 += 正则梯度
            // ∂/∂w (λw²) = 2λw
            gradWeightList[weightPoint] += 2 * lambda * weightList[weightPoint];
        }
    }


    /**
     * 使用答案构造向量，用于反向传播
     *
     * @return
     */
    private double[] answerConstructionVector() {
        double[] vector = new double[outputLayerSize];
        vector[answer] = 1;
        return vector;
    }

    /**
     * 打印出来所有的权重数据和偏置数
     */
    public void printAllData() {
        System.out.println("=======================================");
        System.out.println("输入层大小：" + this.inputLayerSize);
        System.out.println("隐藏层大小：" + this.hiddenLayerSize);
        System.out.println("输出层大小：" + this.outputLayerSize);
        System.out.println("学习率：" + this.learningRate);
        System.out.println("权重1：" + Arrays.toString(this.inputAndHiddenWeightLayer));
        System.out.println("偏置1：" + Arrays.toString(this.hiddenLayerBiasArray));
        System.out.println("权重2：" + Arrays.toString(this.hideAndOutputWeightLayer));
        System.out.println("偏置2：" + Arrays.toString(this.outputLayerBiasArray));
    }

    public void setInputLayerSize(int inputLayerSize) {
        this.inputLayerSize = inputLayerSize;
    }

    public void setHiddenLayerSize(int hiddenLayerSize) {
        this.hiddenLayerSize = hiddenLayerSize;
    }

    public void setOutputLayerSize(int outputLayerSize) {
        this.outputLayerSize = outputLayerSize;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public void setLambda(double lambda) {
        this.lambda = lambda;
    }
}
