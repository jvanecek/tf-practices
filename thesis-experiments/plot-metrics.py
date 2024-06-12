from os import listdir
from os.path import isfile, join
import tensorflow as tf
import matplotlib.pyplot as plt

from google.protobuf.json_format import MessageToDict


def find_first_file_in(experiment_filename, platform_name, stage_name):
  path = f'{experiment_filename}/{platform_name}/{stage_name}'
  print(path)
  return f'{path}/{listdir(path)[0]}'

def event_simple_value_decoder(event_value):
  event_as_dict = MessageToDict(event_value)
  return event_as_dict['simpleValue']

def event_tensor_value_decoder(event_value):
  decoded_tensor = tf.io.decode_raw( event_value.tensor.tensor_content , tf.float32)
  return decoded_tensor.numpy()[0]

def decode_event(record_bytes, metrics, value_decoder):

    event = tf.compat.v1.Event.FromString(record_bytes.numpy())
    if( 'summary' in MessageToDict(event).keys() ):

      event_value = event.summary.value[0]
      if( event_value.tag in metrics.keys() ):

        if( event_value.tag == 'epoch_loss'):
          metrics['wall_time'].append( event.wall_time )
          metrics['time'].append( event.wall_time - metrics['wall_time'][0] )

        metrics[event_value.tag].append( value_decoder(event_value) )


def parse_record_file(file_name, value_decoder):

  parsed_metrics = {
    'epoch_loss' : [],
    'epoch_sparse_categorical_accuracy' : [],
    'epoch_steps_per_second' : [],
    'wall_time' : [],
    'time' : []
  }

  for raw_record in tf.data.TFRecordDataset(file_name):
    decode_event(raw_record, parsed_metrics, value_decoder)

  return parsed_metrics

def same_stage_plot(metrics_collected, metric_names, stage_name, save_path):

    markers = {
        'pharo' : 'x',
        'vast' : '+',
        'python' : 'o'
    }
    x_values = range(0, 10)

    plt.figure(figsize=(12, 6))

    for metric_name in metric_names:
        plt.subplot(1, 2, metric_names.index(metric_name) + 1)  # Create subplots side by side
        for platform in metrics_collected.keys():
            plt.plot(x_values, metrics_collected[platform][metric_name], label=platform, marker=markers[platform])

        plt.xlabel('epoch')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} during {stage_name}')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()  # Adjust spacing between subplots
    # Save the figure instead of showing it
    plt.savefig(save_path)
    plt.close()


def parse_tensorboard_logs(logs_path):
    def _parse_record_file(logs_path, platform, stage, value_decoder):
        full_path = find_first_file_in(logs_path, platform,  stage)
        return parse_record_file(full_path, value_decoder)

    return (
        _parse_record_file(logs_path, 'pharo',  'train',      event_simple_value_decoder),
        _parse_record_file(logs_path, 'pharo',  'validation', event_simple_value_decoder),
        _parse_record_file(logs_path, 'vast',   'train',      event_simple_value_decoder),
        _parse_record_file(logs_path, 'vast',   'validation', event_simple_value_decoder),
        _parse_record_file(logs_path, 'python', 'train',      event_tensor_value_decoder),
        _parse_record_file(logs_path, 'python', 'validation', event_tensor_value_decoder)
    )


experimentsPath = './logs/experiment-3'

(
    pharo_train_metrics,
    pharo_val_metrics,
    vast_train_metrics,
    vast_val_metrics,
    python_train_metrics,
    python_val_metrics
) = parse_tensorboard_logs(experimentsPath)


same_stage_plot(
    metrics_collected={ 'pharo' : pharo_train_metrics, 'python' : python_train_metrics, 'vast' : vast_train_metrics },
    metric_names=['epoch_loss', 'epoch_sparse_categorical_accuracy'],
    stage_name='training',
    save_path=experimentsPath+'/training-metrics.png')

same_stage_plot(
    metrics_collected={ 'pharo' : pharo_train_metrics, 'python' : python_train_metrics, 'vast' : vast_train_metrics },
    metric_names=['time'],
    stage_name='training',
    save_path=experimentsPath+'/training-time.png')

same_stage_plot(
    metrics_collected={ 'pharo' : pharo_val_metrics, 'python' : python_val_metrics, 'vast' : vast_val_metrics },
    metric_names=['epoch_loss', 'epoch_sparse_categorical_accuracy'],
    stage_name='validation',
    save_path=experimentsPath+'/validation-metrics.png')

same_stage_plot(
    metrics_collected={ 'pharo' : pharo_val_metrics, 'python' : python_val_metrics, 'vast' : vast_val_metrics },
    metric_names=['time'],
    stage_name='validation',
    save_path=experimentsPath+'/validation-time.png')
