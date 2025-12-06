package ru.mcashesha.kmeans;

import java.util.Arrays;
import java.util.Random;
import ru.mcashesha.metrics.Metric;

class HierarchicalKMeans implements KMeans<HierarchicalKMeans.Result> {

    private final int branchFactor;
    private final int maxDepth;
    private final int minClusterSize;
    private final int maxIterationsPerLevel;
    private final float tolerance;
    private final Metric.Type metricType;
    private final Metric.Engine metricEngine;
    private final Random random;

    public HierarchicalKMeans(int branchFactor,
        int maxDepth,
        int minClusterSize,
        int maxIterationsPerLevel,
        float tolerance,
        Random random,
        Metric.Type metricType,
        Metric.Engine metricEngine) {

        if (branchFactor <= 1)
            throw new IllegalArgumentException("branchFactor must be >= 2");
        if (maxDepth <= 0)
            throw new IllegalArgumentException("maxDepth must be > 0");
        if (minClusterSize <= 0)
            throw new IllegalArgumentException("minClusterSize must be > 0");
        if (maxIterationsPerLevel <= 0)
            throw new IllegalArgumentException("maxIterationsPerLevel must be > 0");
        if (tolerance < 0.0f)
            throw new IllegalArgumentException("tolerance must be >= 0");
        if (metricType == null || metricEngine == null)
            throw new IllegalArgumentException("metricType and metricEngine must be non-null");
        if (random == null)
            throw new IllegalArgumentException("random must be non-null");

        this.branchFactor = branchFactor;
        this.maxDepth = maxDepth;
        this.minClusterSize = minClusterSize;
        this.maxIterationsPerLevel = maxIterationsPerLevel;
        this.tolerance = tolerance;
        this.random = random;
        this.metricType = metricType;
        this.metricEngine = metricEngine;
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

        int[] allIndices = new int[sampleCnt];
        for (int i = 0; i < sampleCnt; i++)
            allIndices[i] = i;

        Node root = buildNode(data, allIndices, 0, dimension);

        int leafCnt = countLeaves(root);

        float[][] leafCentroids = new float[leafCnt][];
        int[] leafAssignments = new int[sampleCnt];

        IntWrapper leafIdCounter = new IntWrapper();
        assignLeafIdsAndFill(root, leafCentroids, leafAssignments, leafIdCounter);

        float loss = computeLoss(data, leafAssignments, leafCentroids);

        int[] clusterSizes = computeClusterSizes(leafAssignments, leafCnt);

        return new Result(root, leafAssignments, leafCentroids, loss, clusterSizes);
    }

    @Override public int[] predict(float[][] data, Result model) {
        if (data == null || data.length == 0)
            throw new IllegalArgumentException("data must be non-null and non-empty");
        if (model == null)
            throw new IllegalArgumentException("model must be non-null");
        if (model.getRoot() == null)
            throw new IllegalArgumentException("model root must be non-null");

        int dimension = validateAndGetDimension(data);
        if (model.getRoot().getCentroid().length != dimension)
            throw new IllegalArgumentException("data dimension must match tree centroid dimension");

        int[] labels = new int[data.length];

        for (int i = 0; i < data.length; i++) {
            float[] point = data[i];
            Node node = model.getRoot();

            while (!node.isLeaf()) {
                Node[] children = node.getChildren();
                if (children == null || children.length == 0)
                    break;

                Node bestChild = null;
                float bestDistance = Float.POSITIVE_INFINITY;

                for (Node child : children) {
                    float distance = metricType.distance(metricEngine, point, child.centroid);
                    if (distance < bestDistance) {
                        bestDistance = distance;
                        bestChild = child;
                    }
                }

                if (bestChild == null)
                    break;

                node = bestChild;
            }

            if (node.getLeafId() < 0)
                throw new IllegalStateException("Leaf node has no leafId assigned");

            labels[i] = node.getLeafId();
        }

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

    private Node buildNode(float[][] data,
        int[] indices,
        int level,
        int dimension) {
        int sampleCnt = indices.length;

        float[] centroid = computeCentroid(data, indices, dimension);

        if (level >= maxDepth - 1 || sampleCnt < minClusterSize)
            return new Node(level, centroid, null, indices);

        int locClusterCnt = Math.min(branchFactor, sampleCnt);
        if (locClusterCnt < 2)
            return new Node(level, centroid, null, indices);

        float[][] subset = new float[sampleCnt][];
        for (int i = 0; i < sampleCnt; i++)
            subset[i] = data[indices[i]];

        LloydKMeans kmeans = new LloydKMeans(
            locClusterCnt,
            metricType,
            metricEngine,
            maxIterationsPerLevel,
            tolerance,
            random
        );

        LloydKMeans.Result kmResult = kmeans.fit(subset);
        int[] labels = kmResult.getClusterAssignments();

        int[] clusterSizes = new int[locClusterCnt];
        for (int label : labels) {
            if (label < 0 || label >= locClusterCnt)
                throw new IllegalStateException("KMeans produced invalid label: " + label);
            clusterSizes[label]++;
        }

        int nonEmptyClusterCnt = 0;
        for (int c = 0; c < locClusterCnt; c++) {
            if (clusterSizes[c] > 0)
                nonEmptyClusterCnt++;
        }

        if (nonEmptyClusterCnt <= 1)
            return new Node(level, centroid, null, indices);

        int[] clusterIdToChildIdx = new int[locClusterCnt];
        Arrays.fill(clusterIdToChildIdx, -1);

        int childIdx = 0;
        for (int c = 0; c < locClusterCnt; c++) {
            if (clusterSizes[c] > 0) {
                clusterIdToChildIdx[c] = childIdx;
                childIdx++;
            }
        }

        int[][] childIndices = new int[nonEmptyClusterCnt][];
        int[] offsets = new int[nonEmptyClusterCnt];
        for (int i = 0; i < nonEmptyClusterCnt; i++)
            childIndices[i] = new int[0];

        int[] childSizes = new int[nonEmptyClusterCnt];
        for (int c = 0; c < locClusterCnt; c++) {
            int mappedChildIdx = clusterIdToChildIdx[c];
            if (mappedChildIdx >= 0)
                childSizes[mappedChildIdx] = clusterSizes[c];
        }

        for (int i = 0; i < nonEmptyClusterCnt; i++)
            childIndices[i] = new int[childSizes[i]];

        Arrays.fill(offsets, 0);
        for (int i = 0; i < sampleCnt; i++) {
            int originalCluster = labels[i];
            int mappedChild = clusterIdToChildIdx[originalCluster];
            int pos = offsets[mappedChild]++;
            childIndices[mappedChild][pos] = indices[i];
        }

        Node[] children = new Node[nonEmptyClusterCnt];
        for (int i = 0; i < nonEmptyClusterCnt; i++)
            children[i] = buildNode(data, childIndices[i], level + 1, dimension);

        return new Node(level, centroid, children, null);
    }

    private float computeLoss(float[][] data,
        int[] leafAssignments,
        float[][] leafCentroids) {
        float loss = 0.0f;

        for (int i = 0; i < data.length; i++) {
            int clusterIdx = leafAssignments[i];
            float[] point = data[i];
            float[] centroid = leafCentroids[clusterIdx];

            float distance = metricType.distance(metricEngine, point, centroid);
            loss += distance;
        }

        return loss;
    }

    private float[] computeCentroid(float[][] data,
        int[] indices,
        int dimension) {
        float[] centroid = new float[dimension];
        int cnt = indices.length;
        if (cnt == 0)
            return centroid;

        for (int idx : indices) {
            float[] point = data[idx];
            for (int d = 0; d < dimension; d++)
                centroid[d] += point[d];
        }

        float invCnt = 1.0f / (float)cnt;
        for (int d = 0; d < dimension; d++)
            centroid[d] *= invCnt;

        return centroid;
    }

    private int countLeaves(Node node) {
        if (node == null)
            return 0;
        if (node.isLeaf())
            return 1;

        int cnt = 0;
        Node[] children = node.getChildren();
        if (children != null) {
            for (Node child : children)
                cnt += countLeaves(child);
        }
        return cnt;
    }

    private void assignLeafIdsAndFill(Node node,
        float[][] leafCentroids,
        int[] leafAssignments,
        IntWrapper leafIdCounter) {
        if (node == null)
            return;

        if (node.isLeaf()) {
            int leafId = leafIdCounter.val++;
            node.leafId = leafId;

            leafCentroids[leafId] = Arrays.copyOf(node.centroid, node.centroid.length);

            if (node.pointIndices != null) {
                for (int idx : node.pointIndices)
                    leafAssignments[idx] = leafId;
            }
        }
        else {
            Node[] children = node.getChildren();
            if (children != null) {
                for (Node child : children)
                    assignLeafIdsAndFill(child, leafCentroids, leafAssignments, leafIdCounter);
            }
        }
    }

    private int[] computeClusterSizes(int[] assignments, int clusterCnt) {
        int[] sizes = new int[clusterCnt];

        for (int label : assignments) {
            if (label < 0 || label >= clusterCnt) {
                throw new IllegalStateException(
                    "invalid cluster label " + label + " (expected 0.." + (clusterCnt - 1) + ")"
                );
            }
            sizes[label]++;
        }

        return sizes;
    }

    private static final class IntWrapper {
        int val;
    }

    public static final class Node {
        private final int level;
        private final float[] centroid;
        private final Node[] children;
        private final int[] pointIndices;
        private int leafId = -1;

        Node(int level,
            float[] centroid,
            Node[] children,
            int[] pointIndices) {
            this.level = level;
            this.centroid = centroid;
            this.children = children;
            this.pointIndices = pointIndices;
        }

        public int getLevel() {
            return level;
        }

        public float[] getCentroid() {
            return centroid;
        }

        public Node[] getChildren() {
            return children;
        }

        public int[] getPointIndices() {
            return pointIndices;
        }

        public int getLeafId() {
            return leafId;
        }

        public boolean isLeaf() {
            return children == null || children.length == 0;
        }
    }

    static class Result implements ClusteringResult {
        private final Node root;
        private final int[] leafAssignments;
        private final float[][] leafCentroids;
        private final float loss;
        private final int[] clusterSizes;

        Result(Node root,
            int[] leafAssignments,
            float[][] leafCentroids,
            float loss,
            int[] clusterSizes) {
            this.root = root;
            this.leafAssignments = leafAssignments;
            this.leafCentroids = leafCentroids;
            this.loss = loss;
            this.clusterSizes = clusterSizes;
        }

        public Node getRoot() {
            return root;
        }

        @Override public int[] getClusterAssignments() {
            return leafAssignments;
        }

        @Override public float[][] getCentroids() {
            return leafCentroids;
        }

        @Override public float getLoss() {
            return loss;
        }

        @Override public int[] getClusterSizes() {
            return clusterSizes;
        }
    }
}
