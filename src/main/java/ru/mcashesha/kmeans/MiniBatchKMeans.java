package ru.mcashesha.kmeans;

import java.util.Arrays;
import java.util.Random;
import ru.mcashesha.metrics.Metric;

class MiniBatchKMeans implements KMeans<MiniBatchKMeans.Result> {

    private final int clusterCnt;
    private final int batchSize;
    private final int maxIterations;
    private final int maxNoImprovementIterations;
    private final float tolerance;
    private final Metric.Type metricType;
    private final Metric.Engine metricEngine;
    private final Random random;

    public MiniBatchKMeans(
        int clusterCnt,
        int batchSize,
        Metric.Type metricType,
        Metric.Engine metricEngine,
        int maxIterations,
        float tolerance,
        int maxNoImprovementIterations,
        Random random) {
        if (clusterCnt <= 0)
            throw new IllegalArgumentException("clusterCount must be > 0");
        if (batchSize <= 0)
            throw new IllegalArgumentException("batchSize must be > 0");
        if (metricType == null || metricEngine == null)
            throw new IllegalArgumentException("metricType and metricEngine must be non-null");
        if (maxIterations <= 0)
            throw new IllegalArgumentException("maxIterations must be > 0");
        if (maxNoImprovementIterations <= 0)
            throw new IllegalArgumentException("maxNoImprovementIterations must be > 0");
        if (tolerance < 0.0f)
            throw new IllegalArgumentException("tolerance must be >= 0");
        if (random == null)
            throw new IllegalArgumentException("random must be non-null");

        this.clusterCnt = clusterCnt;
        this.batchSize = batchSize;
        this.metricType = metricType;
        this.metricEngine = metricEngine;
        this.maxIterations = maxIterations;
        this.tolerance = tolerance;
        this.maxNoImprovementIterations = maxNoImprovementIterations;
        this.random = random;
    }

    @Override public Metric.Type getMetricType() {
        return metricType;
    }

    @Override public Metric.Engine getMetricEngine() {
        return metricEngine;
    }

    @Override public Result fit(float[][] data) {
        if (data == null || data.length == 0)
            throw new IllegalArgumentException("data must be non-null and non-empty");

        int sampleCnt = data.length;
        int dimension = validateAndGetDimension(data);

        if (clusterCnt > sampleCnt) {
            throw new IllegalArgumentException(
                "clusterCount (" + clusterCnt + ") must be <= number of samples (" + sampleCnt + ")"
            );
        }

        float[][] centroids = initializeCentroidsKMeansPlusPlus(data, sampleCnt, dimension);

        long[] clusterCounts = new long[clusterCnt];

        int performedIterations = 0;
        float lastAverageBatchLoss = Float.POSITIVE_INFINITY;
        int noImprovementIterations = 0;

        int actualBatchSize = Math.min(batchSize, sampleCnt);
        int[] batchIndices = new int[actualBatchSize];
        float[][] batchSums = new float[clusterCnt][dimension];
        int[] batchClusterCounts = new int[clusterCnt];

        for (int iteration = 0; iteration < maxIterations; iteration++) {
            for (int i = 0; i < actualBatchSize; i++)
                batchIndices[i] = random.nextInt(sampleCnt);

            for (int c = 0; c < clusterCnt; c++) {
                Arrays.fill(batchSums[c], 0.0f);
                batchClusterCounts[c] = 0;
            }

            float batchLossSum = assignMiniBatch(data, centroids, batchIndices, batchSums, batchClusterCounts);
            float averageBatchLoss = batchLossSum / actualBatchSize;

            updateCentroidsFromMiniBatch(centroids, clusterCounts, batchSums, batchClusterCounts);

            performedIterations++;

            if (!Float.isFinite(averageBatchLoss))
                break;

            if (Math.abs(lastAverageBatchLoss - averageBatchLoss) <= tolerance) {
                noImprovementIterations++;
                if (noImprovementIterations >= maxNoImprovementIterations)
                    break;
            }
            else
                noImprovementIterations = 0;

            lastAverageBatchLoss = averageBatchLoss;
        }

        int[] labels = new int[sampleCnt];
        float[] pointErrors = new float[sampleCnt];

        assignPointsToClusters(data, centroids, labels, pointErrors);

        float[][] newCentroids = new float[clusterCnt][dimension];
        int[] clusterSizes = new int[clusterCnt];
        recomputeCentroids(data, labels, newCentroids, clusterSizes);

        boolean changed = handleEmptyClusters(data, newCentroids, clusterSizes, labels, pointErrors);
        if (changed)
            recomputeCentroids(data, labels, newCentroids, clusterSizes);

        for (int c = 0; c < clusterCnt; c++)
            System.arraycopy(newCentroids[c], 0, centroids[c], 0, dimension);

        float finalLoss = assignPointsToClusters(data, centroids, labels, null);

        return new Result(labels, centroids, performedIterations, finalLoss, clusterSizes);
    }

    @Override public int[] predict(float[][] data, Result model) {
        if (data == null || data.length == 0)
            throw new IllegalArgumentException("data must be non-null and non-empty");
        if (model == null)
            throw new IllegalArgumentException("model must be non-null");

        int dimension = validateAndGetDimension(data);
        float[][] centroids = model.centroids;

        if (centroids == null || centroids.length == 0)
            throw new IllegalArgumentException("centroids must be non-null and non-empty");

        if (centroids.length != clusterCnt) {
            throw new IllegalArgumentException(
                "model cluster count (" + centroids.length +
                    ") does not match this KMeans configuration (" + clusterCnt + ")"
            );
        }

        int centroidDimension = centroids[0].length;
        if (centroidDimension == 0)
            throw new IllegalArgumentException("centroids must have positive dimension");
        if (centroidDimension != dimension)
            throw new IllegalArgumentException("dimension mismatch between data and centroids");

        for (int c = 1; c < centroids.length; c++) {
            if (centroids[c] == null || centroids[c].length != centroidDimension)
                throw new IllegalArgumentException("all centroids must be non-null and have the same dimension");
        }

        int[] labels = new int[data.length];
        assignPointsToClusters(data, centroids, labels);
        return labels;
    }

    private int validateAndGetDimension(float[][] data) {
        if (data[0] == null)
            throw new IllegalArgumentException("points must be non-null");

        int dimension = data[0].length;
        if (dimension == 0)
            throw new IllegalArgumentException("points must have positive dimension");

        for (int i = 1; i < data.length; i++) {
            float[] point = data[i];
            if (point == null || point.length != dimension)
                throw new IllegalArgumentException("all points must be non-null and have the same dimension");
        }

        return dimension;
    }

    private float[][] initializeCentroidsKMeansPlusPlus(float[][] data,
        int sampleCnt,
        int dimension) {

        float[][] centroids = new float[clusterCnt][dimension];

        int firstIdx = random.nextInt(sampleCnt);
        System.arraycopy(data[firstIdx], 0, centroids[0], 0, dimension);

        float[] minDistances = new float[sampleCnt];

        for (int i = 0; i < sampleCnt; i++) {
            float dist = metricType.distance(metricEngine, data[i], centroids[0]);
            minDistances[i] = dist;
        }

        for (int c = 1; c < clusterCnt; c++) {
            float totalDistance = 0.0f;

            for (int i = 0; i < sampleCnt; i++)
                totalDistance += minDistances[i];

            int chosenIdx;

            if (totalDistance == 0.0f)
                chosenIdx = random.nextInt(sampleCnt);
            else {
                float r = random.nextFloat() * totalDistance;
                float cumulative = 0.0f;

                chosenIdx = sampleCnt - 1;

                for (int i = 0; i < sampleCnt; i++) {
                    cumulative += minDistances[i];
                    if (cumulative >= r) {
                        chosenIdx = i;
                        break;
                    }
                }
            }

            System.arraycopy(data[chosenIdx], 0, centroids[c], 0, dimension);

            float[] newCentroid = centroids[c];
            for (int i = 0; i < sampleCnt; i++) {
                float dist = metricType.distance(metricEngine, data[i], newCentroid);
                if (dist < minDistances[i])
                    minDistances[i] = dist;
            }
        }

        return centroids;
    }

    private float assignMiniBatch(float[][] data,
        float[][] centroids,
        int[] batchIndices,
        float[][] batchSums,
        int[] batchClusterCounts) {
        int dimension = centroids[0].length;
        float batchLoss = 0.0f;

        for (int sampleIdx : batchIndices) {
            float[] point = data[sampleIdx];

            int nearestClusterIdx = -1;
            float nearestDistance = Float.POSITIVE_INFINITY;

            for (int c = 0; c < clusterCnt; c++) {
                float distance = metricType.distance(metricEngine, point, centroids[c]);
                if (distance < nearestDistance) {
                    nearestDistance = distance;
                    nearestClusterIdx = c;
                }
            }

            batchLoss += nearestDistance;

            float[] sum = batchSums[nearestClusterIdx];
            for (int d = 0; d < dimension; d++)
                sum[d] += point[d];

            batchClusterCounts[nearestClusterIdx]++;
        }

        return batchLoss;
    }

    private void updateCentroidsFromMiniBatch(float[][] centroids,
        long[] clusterCounts,
        float[][] batchSums,
        int[] batchClusterCounts) {
        int dimension = centroids[0].length;

        for (int c = 0; c < clusterCnt; c++) {
            int batchCnt = batchClusterCounts[c];
            if (batchCnt == 0)
                continue;

            long oldCnt = clusterCounts[c];
            long newCnt = oldCnt + batchCnt;

            clusterCounts[c] = newCnt;

            float invNewCnt = 1.0f / newCnt;
            float[] centroid = centroids[c];
            float[] sum = batchSums[c];

            for (int d = 0; d < dimension; d++)
                centroid[d] = (centroid[d] * oldCnt + sum[d]) * invNewCnt;
        }
    }

    private float assignPointsToClusters(float[][] data,
        float[][] centroids,
        int[] labels,
        float[] pointErrors) {
        int sampleCnt = data.length;
        float loss = 0.0f;

        for (int i = 0; i < sampleCnt; i++) {
            float[] point = data[i];

            int nearestClusterIdx = -1;
            float nearestDistance = Float.POSITIVE_INFINITY;

            for (int c = 0; c < clusterCnt; c++) {
                float distance = metricType.distance(metricEngine, point, centroids[c]);
                if (distance < nearestDistance) {
                    nearestDistance = distance;
                    nearestClusterIdx = c;
                }
            }

            labels[i] = nearestClusterIdx;

            loss += nearestDistance;

            if (pointErrors != null)
                pointErrors[i] = nearestDistance;
        }

        return loss;
    }

    private void assignPointsToClusters(float[][] data,
        float[][] centroids,
        int[] labels) {
        assignPointsToClusters(data, centroids, labels, null);
    }

    private void recomputeCentroids(float[][] data,
        int[] labels,
        float[][] newCentroids,
        int[] clusterSizes) {
        int dimension = newCentroids[0].length;

        for (int c = 0; c < clusterCnt; c++) {
            Arrays.fill(newCentroids[c], 0);
            clusterSizes[c] = 0;
        }

        for (int i = 0; i < data.length; i++) {
            int clusterIdx = labels[i];
            clusterSizes[clusterIdx]++;

            float[] centroidSum = newCentroids[clusterIdx];
            float[] point = data[i];

            for (int d = 0; d < dimension; d++)
                centroidSum[d] += point[d];
        }

        for (int c = 0; c < clusterCnt; c++) {
            int size = clusterSizes[c];
            if (size > 0) {
                float invSize = 1.0f / size;
                float[] centroid = newCentroids[c];
                for (int d = 0; d < dimension; d++)
                    centroid[d] *= invSize;
            }
        }
    }

    private boolean handleEmptyClusters(float[][] data,
        float[][] newCentroids,
        int[] clusterSizes,
        int[] labels,
        float[] pointErrors) {
        int sampleCnt = data.length;
        int dimension = newCentroids[0].length;

        boolean changed = false;
        boolean[] taken = new boolean[sampleCnt];

        for (int emptyCluster = 0; emptyCluster < clusterCnt; emptyCluster++) {
            if (clusterSizes[emptyCluster] != 0)
                continue;

            int chosenIdx = choosePointFromLargestCluster(labels, clusterSizes, pointErrors, taken);

            if (chosenIdx == -1)
                chosenIdx = chooseGlobalWorstPoint(labels, pointErrors, taken);

            if (chosenIdx == -1)
                chosenIdx = random.nextInt(sampleCnt);

            taken[chosenIdx] = true;

            int oldCluster = labels[chosenIdx];

            labels[chosenIdx] = emptyCluster;
            clusterSizes[emptyCluster] = 1;
            if (oldCluster >= 0)
                clusterSizes[oldCluster] = Math.max(0, clusterSizes[oldCluster] - 1);

            System.arraycopy(data[chosenIdx], 0, newCentroids[emptyCluster], 0, dimension);

            changed = true;
        }

        return changed;
    }

    private int choosePointFromLargestCluster(int[] labels,
        int[] clusterSizes,
        float[] pointErrors,
        boolean[] taken) {
        int largestCluster = -1;
        int maxSize = 0;
        for (int c = 0; c < clusterCnt; c++) {
            if (clusterSizes[c] > maxSize) {
                maxSize = clusterSizes[c];
                largestCluster = c;
            }
        }
        if (largestCluster < 0 || maxSize <= 1)
            return -1;

        int bestIdx = -1;
        float bestError = -1f;
        for (int i = 0; i < labels.length; i++) {
            if (taken[i])
                continue;
            if (labels[i] != largestCluster)
                continue;
            float err = pointErrors != null ? pointErrors[i] : 1.0f;
            if (err > bestError) {
                bestError = err;
                bestIdx = i;
            }
        }

        return bestIdx;
    }

    private int chooseGlobalWorstPoint(int[] labels,
        float[] pointErrors,
        boolean[] taken) {
        int bestIdx = -1;
        float bestError = -1f;
        for (int i = 0; i < labels.length; i++) {
            if (taken[i])
                continue;
            if (labels[i] < 0)
                continue;
            float err = pointErrors != null ? pointErrors[i] : 1.0f;
            if (err > bestError) {
                bestError = err;
                bestIdx = i;
            }
        }

        return bestIdx;
    }

    static class Result implements KMeans.ClusteringResult {
        private final int[] labels;
        private final float[][] centroids;
        private final int iterations;
        private final float loss;
        private final int[] clusterSizes;

        public Result(int[] labels,
            float[][] centroids,
            int iterations,
            float loss,
            int[] clusterSizes) {
            this.labels = labels;
            this.centroids = centroids;
            this.iterations = iterations;
            this.loss = loss;
            this.clusterSizes = clusterSizes;
        }

        @Override public int[] getClusterAssignments() {
            return labels;
        }

        @Override public float[][] getCentroids() {
            return centroids;
        }

        public int getIterations() {
            return iterations;
        }

        @Override public float getLoss() {
            return loss;
        }

        @Override public int[] getClusterSizes() {
            return clusterSizes;
        }
    }

}
