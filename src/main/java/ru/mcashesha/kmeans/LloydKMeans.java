package ru.mcashesha.kmeans;

import java.util.Arrays;
import java.util.Random;
import ru.mcashesha.metrics.Metric;

class LloydKMeans implements KMeans<LloydKMeans.Result> {
    private final int clusterCnt;
    private final int maxIterations;
    private final float tolerance;
    private final Metric.Type metricType;
    private final Metric.Engine metricEngine;
    private final Random random;

    public LloydKMeans(int clusterCnt,
        Metric.Type metricType,
        Metric.Engine metricEngine,
        int maxIterations,
        float tolerance,
        Random random) {
        if (clusterCnt <= 0)
            throw new IllegalArgumentException("clusterCount must be > 0");
        if (metricType == null || metricEngine == null)
            throw new IllegalArgumentException("metricType and metricEngine must be non-null");
        if (maxIterations <= 0)
            throw new IllegalArgumentException("maxIterations must be > 0");
        if (tolerance < 0)
            throw new IllegalArgumentException("tolerance must be >= 0");
        if (random == null)
            throw new IllegalArgumentException("random must be non-null");

        this.clusterCnt = clusterCnt;
        this.metricType = metricType;
        this.metricEngine = metricEngine;
        this.maxIterations = maxIterations;
        this.tolerance = tolerance;
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

        int[] labels = new int[sampleCnt];
        Arrays.fill(labels, -1);

        float[][] newCentroids = new float[clusterCnt][dimension];
        int[] clusterSizes = new int[clusterCnt];

        float[] pointErrors = new float[sampleCnt];

        int performedIterations = 0;

        for (int iteration = 0; iteration < maxIterations; iteration++) {
            assignPointsToClusters(data, centroids, labels, pointErrors);

            recomputeCentroids(data, labels, newCentroids, clusterSizes);

            boolean changed = handleEmptyClusters(data, newCentroids, clusterSizes, labels, pointErrors);
            if (changed)
                recomputeCentroids(data, labels, newCentroids, clusterSizes);

            float maxShift = computeMaxCentroidShift(centroids, newCentroids);

            for (int c = 0; c < clusterCnt; c++)
                System.arraycopy(newCentroids[c], 0, centroids[c], 0, dimension);

            performedIterations = iteration + 1;

            if (maxShift <= tolerance)
                break;
        }

        float finalLoss = assignPointsToClusters(data, centroids, labels, null);
        int[] finalClusterSizes = computeClusterSizes(labels, clusterCnt);

        return new Result(labels, centroids, performedIterations, finalLoss, finalClusterSizes);
    }

    @Override public int[] predict(float[][] data, Result model) {
        if (data == null || data.length == 0)
            throw new IllegalArgumentException("data must be non-null and non-empty");
        if (model == null)
            throw new IllegalArgumentException("model must be non-null");

        int dimension = validateAndGetDimension(data);

        float[][] centroids = model.centroids;
        if (centroids == null || centroids.length == 0)
            throw new IllegalArgumentException("model must contain at least one centroid");

        if (centroids.length != clusterCnt) {
            throw new IllegalArgumentException(
                "model cluster count (" + centroids.length +
                    ") does not match this KMeans configuration (" + clusterCnt + ")"
            );
        }

        int centroidDim = centroids[0].length;
        if (centroidDim == 0)
            throw new IllegalArgumentException("centroids must have positive dimension");
        if (centroidDim != dimension)
            throw new IllegalArgumentException("dimension mismatch between data and centroids");

        for (int c = 1; c < centroids.length; c++) {
            if (centroids[c] == null || centroids[c].length != centroidDim)
                throw new IllegalArgumentException("all centroids must be non-null and have the same dimension");
        }

        int[] labels = new int[data.length];

        assignPointsToClusters(data, centroids, labels, null);

        return labels;
    }

    private int[] computeClusterSizes(int[] labels, int clusterCnt) {
        int[] sizes = new int[clusterCnt];

        for (int label : labels) {
            if (label < 0 || label >= clusterCnt) {
                throw new IllegalStateException(
                    "invalid cluster label " + label + " (expected 0.." + (clusterCnt - 1) + ")"
                );
            }
            sizes[label]++;
        }

        return sizes;
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
            float distance = metricType.distance(metricEngine, data[i], centroids[0]);
            minDistances[i] = distance;
        }

        for (int c = 1; c < clusterCnt; c++) {
            float totalWeight = 0f;

            for (int i = 0; i < sampleCnt; i++)
                totalWeight += minDistances[i];

            int chosenIdx;

            if (totalWeight == 0f)
                chosenIdx = random.nextInt(sampleCnt);
            else {
                float threshold = random.nextFloat() * totalWeight;
                float cumulative = 0f;
                chosenIdx = sampleCnt - 1;

                for (int i = 0; i < sampleCnt; i++) {
                    cumulative += minDistances[i];
                    if (cumulative >= threshold) {
                        chosenIdx = i;
                        break;
                    }
                }
            }

            System.arraycopy(data[chosenIdx], 0, centroids[c], 0, dimension);

            float[] newCentroid = centroids[c];
            for (int i = 0; i < sampleCnt; i++) {
                float distance = metricType.distance(metricEngine, data[i], newCentroid);
                if (distance < minDistances[i])
                    minDistances[i] = distance;
            }
        }

        return centroids;
    }

    private float assignPointsToClusters(float[][] data,
        float[][] centroids,
        int[] labels,
        float[] pointErrors) {
        int sampleCnt = data.length;
        float loss = 0f;

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
            if (labels[i] == largestCluster) {
                float err = pointErrors != null ? pointErrors[i] : 1.0f;
                if (err > bestError) {
                    bestError = err;
                    bestIdx = i;
                }
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

    private float computeMaxCentroidShift(float[][] oldCentroids,
        float[][] newCentroids) {
        float maxShift = 0;

        for (int c = 0; c < clusterCnt; c++) {
            float shift = metricType.distance(metricEngine, oldCentroids[c], newCentroids[c]);
            if (shift > maxShift)
                maxShift = shift;
        }

        return maxShift;
    }

    static final class Result implements ClusteringResult {
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
