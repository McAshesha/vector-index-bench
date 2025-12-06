package ru.mcashesha.metrics;

public interface Metric {

    float l2Distance(float[] a, float[] b);

    float dotProduct(float[] a, float[] b);

    float cosineDistance(float[] a, float[] b);

    long hammingDistanceB8(byte[] a, byte[] b);

    enum Type {
        L2SQ_DISTANCE() {
            @Override public float distance(Engine engine, float[] a, float[] b) {
                return engine.getMetric().l2Distance(a, b);
            }
        },
        DOT_PRODUCT {
            @Override public float distance(Engine engine, float[] a, float[] b) {
                return engine.getMetric().dotProduct(a, b);
            }
        },
        COSINE_DISTANCE {
            @Override public float distance(Engine engine, float[] a, float[] b) {
                return engine.getMetric().cosineDistance(a, b);
            }
        };

        public abstract float distance(Engine engine, float[] a, float[] b);
    }

    enum Engine {
        SCALAR(new Scalar()),
        VECTOR_API(new VectorAPI()),
        SIMSIMD(new SimSIMD());

        final Metric metric;

        Engine(Metric metric) {
            this.metric = metric;
        }

        Metric getMetric() {
            return metric;
        }
    }

}
