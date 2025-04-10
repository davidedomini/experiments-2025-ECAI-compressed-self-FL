package it.unibo.alchemist.model.monitors

import it.unibo.alchemist.boundary.OutputMonitor
import it.unibo.alchemist.model.layers.Dataset
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.model.{Environment, Node, Position, Time}
import it.unibo.scafi.Sensors
import it.unibo.alchemist.exporter.TestDataExporter
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.PyQuote

class DistributedTestSetEvaluation[P <: Position[P]](
    seed: Double,
    epochs: Int,
    aggregateLocalEvery: Int,
    areas: Int,
    dataShuffle: Boolean,
    lossThreshold: Double,
    sparsityLevel: Double
) extends TestSetEvaluation[P](seed) {

  override def finished(
      environment: Environment[Any, P],
      time: Time,
      step: Long
  ): Unit = {
    val layer =
      environment.getLayer(new SimpleMolecule(Sensors.testsetPhenomena)).get()
    val accuracies =
      nodes(environment)
        .map(node => {
          val weights = node
            .getConcentration(new SimpleMolecule(Sensors.sharedModel))
            .asInstanceOf[py.Dynamic]
          val data = layer
            .getValue(environment.getPosition(node))
            .asInstanceOf[Dataset]
          (weights, data.trainingData)
        })
        .map { case (w, d) => evaluate(w, d) }
    TestDataExporter.CSVExport(
      accuracies,
      s"data-test/test_accuracy_seed-${seed}_epochs-${epochs}" +
        s"_aggregateLocalEvery-${aggregateLocalEvery}_areas-${areas}" +
        s"_batchSize-${batch_size}_dataShuffle-${dataShuffle}" +
        s"_sparsity-${sparsityLevel}_lossThreshold-${lossThreshold}"
    )
    cleanPythonObjects(environment)
  }

}
