package ru.mcashesha.metrics;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

class VectorAPI implements Metric {

    static final VectorSpecies<Float> floatSpecies = FloatVector.SPECIES_PREFERRED;

    static final VectorSpecies<Byte> byteSpecies = ByteVector.SPECIES_PREFERRED;

    @Override public float l2Distance(float[] a, float[] b) {
        FloatVector vectorSumSquares = FloatVector.zero(floatSpecies);

        int index = 0;

        int upperBound = floatSpecies.loopBound(a.length);

        for (; index < upperBound; index += floatSpecies.length()) {
            FloatVector vectorA = FloatVector.fromArray(floatSpecies, a, index);

            FloatVector vectorB = FloatVector.fromArray(floatSpecies, b, index);

            FloatVector vectorDiff = vectorA.sub(vectorB);

            vectorSumSquares = vectorDiff.fma(vectorDiff, vectorSumSquares);
        }

        float sumSquares = vectorSumSquares.reduceLanes(VectorOperators.ADD);

        for (; index < a.length; index++) {
            float diff = a[index] - b[index];

            sumSquares += diff * diff;
        }

        return sumSquares;
    }

    @Override public float dotProduct(float[] a, float[] b) {
        float sum = 0;

        int i = 0;

        int upperBound = floatSpecies.loopBound(a.length);

        for (; i < upperBound; i += floatSpecies.length()) {
            FloatVector va = FloatVector.fromArray(floatSpecies, a, i);

            FloatVector vb = FloatVector.fromArray(floatSpecies, b, i);

            float partialDot = va.mul(vb).reduceLanes(VectorOperators.ADD);

            sum += partialDot;
        }

        for (; i < a.length; i++)
            sum += a[i] * b[i];

        return sum;
    }

    @Override public float cosineDistance(float[] a, float[] b) {
        float dot = 0, sumA = 0, sumB = 0;

        int i = 0, bound = floatSpecies.loopBound(a.length);

        for (; i < bound; i += floatSpecies.length()) {
            FloatVector va = FloatVector.fromArray(floatSpecies, a, i);

            FloatVector vb = FloatVector.fromArray(floatSpecies, b, i);

            dot += va.mul(vb).reduceLanes(VectorOperators.ADD);

            sumA += va.mul(va).reduceLanes(VectorOperators.ADD);

            sumB += vb.mul(vb).reduceLanes(VectorOperators.ADD);
        }

        for (; i < a.length; i++) {
            dot += a[i] * b[i];

            sumA += a[i] * a[i];

            sumB += b[i] * b[i];
        }

        return 1 - (float)(dot / (Math.sqrt(sumA) * Math.sqrt(sumB)));

    }

    @Override public long hammingDistanceB8(byte[] a, byte[] b) {
        long distance = 0;

        int index = 0;

        int upperBound = byteSpecies.loopBound(a.length);

        for (; index < upperBound; index += byteSpecies.length()) {
            ByteVector vectorA = ByteVector.fromArray(byteSpecies, a, index);

            ByteVector vectorB = ByteVector.fromArray(byteSpecies, b, index);

            ByteVector vectorXor = vectorA.lanewise(VectorOperators.XOR, vectorB);

            for (int lane = 0; lane < byteSpecies.length(); lane++) {
                int xorValue = vectorXor.lane(lane);

                distance += Integer.bitCount(xorValue);
            }
        }

        for (; index < a.length; index++) {
            int xorValue = a[index] ^ b[index];

            distance += Integer.bitCount(xorValue);
        }

        return distance;
    }

}
