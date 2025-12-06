package ru.mcashesha.ivf;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.PriorityQueue;
import ru.mcashesha.kmeans.KMeans;
import ru.mcashesha.metrics.Metric;

public class IVFIndexFlat implements IVFIndex {
    private final KMeans<? extends KMeans.ClusteringResult> kMeans;

    private float[][] centroids;
    private List<IntArrayList> invertedLists;

    private float[][] data;
    private int[] ids;
    private int dimension;
    private boolean built;

    public IVFIndexFlat(KMeans<? extends KMeans.ClusteringResult> kMeans) {
        if (kMeans == null)
            throw new IllegalArgumentException("kMeans must be non-null");

        this.kMeans = kMeans;
    }

    private static int[] selectTopNProbeClusters(float[] distances, int nProbe) {
        int clusterCnt = distances.length;
        nProbe = Math.min(nProbe, clusterCnt);

        int[] result = new int[nProbe];
        boolean[] used = new boolean[clusterCnt];

        for (int i = 0; i < nProbe; i++) {
            int bestIdx = -1;
            float bestDistance = Float.POSITIVE_INFINITY;

            for (int c = 0; c < clusterCnt; c++) {
                if (used[c])
                    continue;
                float d = distances[c];
                if (d < bestDistance) {
                    bestDistance = d;
                    bestIdx = c;
                }
            }

            result[i] = bestIdx;
            used[bestIdx] = true;
        }

        return result;
    }

    @Override public void build(float[][] vectors) {
        build(vectors, null);
    }

    @Override public void build(float[][] vectors, int[] ids) {
        if (vectors == null || vectors.length == 0)
            throw new IllegalArgumentException("vectors must be non-empty");
        if (vectors[0] == null)
            throw new IllegalArgumentException("vectors[0] must be non-null");

        int locDimension = vectors[0].length;
        if (locDimension == 0)
            throw new IllegalArgumentException("vector dimension must be > 0");

        for (int i = 1; i < vectors.length; i++) {
            if (vectors[i] == null || vectors[i].length != locDimension) {
                throw new IllegalArgumentException(
                    "all vectors must be non-null and have the same dimension"
                );
            }
        }

        this.dimension = locDimension;
        this.data = vectors;

        if (ids != null) {
            if (ids.length != vectors.length)
                throw new IllegalArgumentException("ids length must match vectors length");
            this.ids = ids;
        }
        else {
            this.ids = new int[vectors.length];
            for (int i = 0; i < this.ids.length; i++)
                this.ids[i] = i;
        }

        KMeans.ClusteringResult clusteringResult = kMeans.fit(vectors);

        this.centroids = clusteringResult.getCentroids();
        int[] sizes = clusteringResult.getClusterSizes();
        int[] assignments = clusteringResult.getClusterAssignments();

        if (centroids == null || centroids.length == 0)
            throw new IllegalStateException("KMeans returned empty centroids");
        if (assignments == null || assignments.length != vectors.length)
            throw new IllegalStateException("KMeans returned inconsistent assignments");

        int clusterCnt = centroids.length;

        for (float[] centroid : centroids) {
            if (centroid == null || centroid.length != dimension)
                throw new IllegalStateException("centroid dimension mismatch");
        }

        this.invertedLists = new ArrayList<>(clusterCnt);

        int totalSize = 0;
        for (int c = 0; c < clusterCnt; c++) {
            int sizeForCluster = sizes[c];
            if (sizeForCluster < 0)
                throw new IllegalStateException("KMeans returned negative cluster size for cluster " + c);
            totalSize += sizeForCluster;
        }
        if (totalSize != vectors.length)
            throw new IllegalStateException(
                "Sum of clusterSizes (" + totalSize + ") != number of vectors (" + vectors.length + ')'
            );

        for (int c = 0; c < clusterCnt; c++)
            this.invertedLists.add(new IntArrayList(sizes[c]));

        for (int i = 0; i < assignments.length; i++) {
            int clusterId = assignments[i];
            if (clusterId < 0 || clusterId >= clusterCnt)
                continue;
            invertedLists.get(clusterId).add(i);
        }

        this.built = true;
    }

    @Override public Metric.Type getMetricType() {
        return kMeans.getMetricType();
    }

    @Override public Metric.Engine getMetricEngine() {
        return kMeans.getMetricEngine();
    }

    @Override public int getCountClusters() {
        return centroids.length;
    }

    @Override public List<SearchResult> search(float[] qry, int topK, int nProbe) {
        if (!built)
            throw new IllegalStateException("Index is not built yet");
        if (qry == null || qry.length != dimension)
            throw new IllegalArgumentException("query must be non-null and match index dimension");
        if (topK <= 0)
            throw new IllegalArgumentException("topK must be > 0");

        int clusterCnt = centroids.length;
        if (clusterCnt == 0)
            return Collections.emptyList();

        nProbe = Math.max(1, Math.min(nProbe, clusterCnt));

        Metric.Type metricType = kMeans.getMetricType();
        Metric.Engine metricEngine = kMeans.getMetricEngine();

        float[] centroidDistances = new float[clusterCnt];
        for (int c = 0; c < clusterCnt; c++)
            centroidDistances[c] = metricType.distance(metricEngine, qry, centroids[c]);

        int[] selectedClusters = selectTopNProbeClusters(centroidDistances, nProbe);

        PriorityQueue<SearchResult> heap = new PriorityQueue<>(
            topK,
            (a, b) -> Float.compare(b.distance, a.distance)
        );

        for (int clusterId : selectedClusters) {
            IntArrayList list = invertedLists.get(clusterId);
            for (int i = 0; i < list.size(); i++) {
                int vectorIdx = list.get(i);
                float d = metricType.distance(metricEngine, qry, data[vectorIdx]);
                int id = ids[vectorIdx];

                if (heap.size() < topK)
                    heap.offer(new SearchResult(id, d, clusterId));
                else if (d < heap.peek().distance) {
                    heap.poll();
                    heap.offer(new SearchResult(id, d, clusterId));
                }
            }
        }

        List<SearchResult> result = new ArrayList<>(heap.size());
        while (!heap.isEmpty())
            result.add(heap.poll());
        Collections.reverse(result);

        return result;
    }

    @Override public int getDimension() {
        return dimension;
    }

    private static final class IntArrayList {
        private int[] data;
        private int size;

        IntArrayList() {
            this.data = new int[16];
            this.size = 0;
        }

        IntArrayList(int size) {
            this.data = new int[Math.min(16, size)];
            this.size = 0;
        }

        void add(int val) {
            if (size == data.length) {
                int[] newData = new int[data.length * 2];
                System.arraycopy(data, 0, newData, 0, data.length);
                data = newData;
            }
            data[size++] = val;
        }

        int get(int idx) {
            if (idx < 0 || idx >= size)
                throw new IndexOutOfBoundsException("index = " + idx + ", size = " + size);
            return data[idx];
        }

        int size() {
            return size;
        }
    }
}
