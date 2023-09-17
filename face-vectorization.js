module.exports = function(RED) {
  function FaceVectorizationNode(config) {
      RED.nodes.createNode(this,config);
      var node = this;

      // ONNX.js를 사용하여 FaceNet 모델 로드
      const onnx = require('onnxjs');
      const fs = require('fs');

      const modelData = fs.readFileSync(config.modelPath);
      const model = onnx.ModelProto.decode(new Uint8Array(modelData));

      // ONNX 런타임 세션 생성
      const session = new onnx.InferenceSession();
      session.prepareModel(model);

      node.on('input', function(msg) {
          // 입력 데이터 가져오기
          const inputData = new onnx.Tensor(new Float32Array(msg.payload), 'float32', [1, 3, 96, 96]);

          // 추론 수행
          const outputData = session.run([inputData]);

          // 결과를 출력으로 설정
          msg.payload = outputData[0].data;

          // 다음 노드로 메시지 전달
          node.send(msg);
      });
  }
  RED.nodes.registerType("face-vectorization",FaceVectorizationNode);
}