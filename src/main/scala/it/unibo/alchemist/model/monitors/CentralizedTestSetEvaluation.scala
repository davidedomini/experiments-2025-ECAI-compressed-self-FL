package it.unibo.alchemist.model.monitors

import it.unibo.alchemist.exporter.TestDataExporter
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.model.{Environment, Position, Time}
import it.unibo.scafi.Sensors
import it.unibo.scafi.interop.PythonModules.utils
import me.shadaj.scalapy.py.{PyQuote, SeqConverters}
import me.shadaj.scalapy.py

class CentralizedTestSetEvaluation[P <: Position[P]](
    seed: Double,
    epochs: Int,
    areas: Int,
    dataShuffle: Boolean,
    sparsityLevel: Double,
) extends TestSetEvaluation[P](seed) {

  override def finished(environment: Environment[Any, P], time: Time, step: Long): Unit = {
    println("Starting evaluation...")
    val weights =
      nodes(environment).head.getConcentration(new SimpleMolecule(Sensors.model)).asInstanceOf[py.Dynamic]
    val data = utils.get_dataset(Seq.empty[Int].toPythonProxy, false, false)
    val accuracy = evaluate(weights, data)
    TestDataExporter.CSVExport(
      List(accuracy),
      s"data-test-baseline/test_accuracy_seed-${seed}_epochs-${epochs}" +
        s"_areas-${areas}_batchSize-${batch_size}_sparsity-${sparsityLevel}_dataShuffle-${dataShuffle}"
    )
    cleanPythonObjects(environment)
  }

}
