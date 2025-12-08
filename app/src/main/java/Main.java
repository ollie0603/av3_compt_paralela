import java.util.Scanner;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.jocl.Pointer;
import org.jocl.*;
import static org.jocl.CL.*;

public class Main {
    private static final String opencl = 
    "__kernel void contador(\n" +
"__global const char* livro,\n" +
"__constant char* palavra,\n" +
"__global int* ocorrencias_grupos,\n" +
"const int tamanho_livro,\n" +
"const int comprimento_busca)\n" +
"{\n" +
"    int id = get_global_id(0);\n" +
"    int id_grupo = get_group_id(0);\n" +
"    int contagem_local = 0;\n" +
"    if (id >= tamanho_livro) return;\n" +

"    // Boundary: must NOT be followed/preceded by a letter (A-Z or a-z)\n" +
"    bool is_prev_letter = (id > 0) && ((livro[id-1] >= 'A' && livro[id-1] <= 'Z') || (livro[id-1] >= 'a' && livro[id-1] <= 'z'));\n" +

"    if (!is_prev_letter && (id + comprimento_busca <= tamanho_livro)) {\n" +
"        bool match = true;\n" +
"        for (int i = 0; i < comprimento_busca; i++) {\n" +
"            char c = livro[id + i];\n" +
"            if (c >= 'A' && c <= 'Z') c += 32; // Lowercase conversion\n" +
"            if (c != palavra[i]) { match = false; break; }\n" +
"        }\n" +
"        if (match) {\n" +
"            int next_idx = id + comprimento_busca;\n" +
"            bool is_next_letter = (next_idx < tamanho_livro) && \n" +
"                ((livro[next_idx] >= 'A' && livro[next_idx] <= 'Z') || (livro[next_idx] >= 'a' && livro[next_idx] <= 'z'));\n" +
"            if (!is_next_letter) contagem_local = 1;\n" +
"        }\n" +
"    }\n" +

"    if (contagem_local > 0) { atomic_add(&ocorrencias_grupos[id_grupo], contagem_local); }\n" +
"}\n";

    public static void main(String[] args) {
        Path path = null;
        Scanner scanner = new Scanner(System.in);
        String livroInput = "";

        while (path == null) {
            System.out.print("Insira o nome do livro: ");
            livroInput = scanner.nextLine().toLowerCase().replaceAll("\\s", "");
            
            if (livroInput.equals("donquixote")) path = Paths.get("DonQuixote-388208.txt");
            else if (livroInput.equals("dracula")) path = Paths.get("Dracula-165307.txt");
            else if (livroInput.equals("mobydick")) path = Paths.get("MobyDick-217452.txt");
            else System.out.println("Livro não encontrado.");
        }

        System.out.print("Insira a palavra que deseja buscar: ");
        String busca = scanner.next().toLowerCase();
        scanner.close();

        try {
            System.out.println("\n--- Serial CPU ---");
            SerialCPU(path, busca);
            System.out.println("\n--- Parallel CPU ---");
            ParallelCPU(path, busca);
            System.out.println("\n--- Parallel GPU/OpenCL ---");
            ParallelGPU(path, busca);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void SerialCPU(Path path, String busca) throws IOException {
    int ocorrencias = 0;
    String conteudo = new String(Files.readAllBytes(path)).toLowerCase();
    int len = busca.length();

    long start = System.currentTimeMillis();
    for (int i = 0; i <= conteudo.length() - len; i++) {
        if (conteudo.substring(i, i + len).equals(busca)) { 
            boolean prevOk = (i == 0) || !Character.isLetter(conteudo.charAt(i - 1));
            boolean nextOk = (i + len == conteudo.length()) || !Character.isLetter(conteudo.charAt(i + len));

            if (prevOk && nextOk) {
                ocorrencias++;
            }
        }
    }
    System.out.println("Ocorrências: " + ocorrencias + " | Tempo: " + (System.currentTimeMillis() - start) + "ms");
} 

    public static void ParallelCPU(Path path, String busca) throws IOException, InterruptedException, ExecutionException {
    String conteudo = new String(Files.readAllBytes(path)).toLowerCase();
    int threads = Runtime.getRuntime().availableProcessors();
    ExecutorService executor = Executors.newFixedThreadPool(threads);
    List<Future<Integer>> futuros = new ArrayList<>();
    
    int totalLen = conteudo.length();
    int chunkSize = (int) Math.ceil((double) totalLen / threads);
    int searchLen = busca.length();

    long start = System.currentTimeMillis();
    for (int i = 0; i < totalLen; i += chunkSize) {
        final int inicio = i;
        final int fim = Math.min(i + chunkSize, totalLen);
        
        futuros.add(executor.submit(() -> {
            int count = 0; 
            for (int j = inicio; j < fim; j++) {
                if (j + searchLen <= totalLen) {
                    if (conteudo.regionMatches(j, busca, 0, searchLen)) {
                        boolean prevOk = (j == 0) || !Character.isLetter(conteudo.charAt(j - 1));
                        boolean nextOk = (j + searchLen == totalLen) || !Character.isLetter(conteudo.charAt(j + searchLen));
                        
                        if (prevOk && nextOk) count++;
                    }
                }
            }
            return count;
        }));
    }

    int total = 0;
    for (Future<Integer> f : futuros) total += f.get();
    executor.shutdown();
    
    long end = System.currentTimeMillis();
    System.out.println("Ocorrências: " + total);
    System.out.println("Tempo: " + (end - start) + "ms");
} 

    public static void ParallelGPU(Path path, String busca) throws IOException {
        byte[] bytes_livro = Files.readAllBytes(path);
        byte[] bytes_palavra = busca.getBytes();
        
        cl_context context = null;
        cl_command_queue queue = null;
        cl_program program = null;
        cl_kernel kernel = null;
        cl_mem livro_buffer = null, palavra_buffer = null, contador_buffer = null;

        try {
            cl_platform_id[] platforms = new cl_platform_id[1];
            clGetPlatformIDs(1, platforms, null);
            
            cl_device_id[] devices = new cl_device_id[1];
            clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 1, devices, null);
            cl_device_id device = devices[0];

            context = clCreateContext(null, 1, new cl_device_id[]{device}, null, null, null);
            queue = clCreateCommandQueueWithProperties(context, device, null, null);
            program = clCreateProgramWithSource(context, 1, new String[]{opencl}, null, null);
            clBuildProgram(program, 0, null, null, null, null);
            kernel = clCreateKernel(program, "contador", null);

            livro_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_char * bytes_livro.length, Pointer.to(bytes_livro), null);
            palavra_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_char * bytes_palavra.length, Pointer.to(bytes_palavra), null);

            int localSize = 256;
            int globalSize = ((bytes_livro.length + localSize - 1) / localSize) * localSize;
            int numGrupos = globalSize / localSize;
            int[] inicio_contador = new int[numGrupos];
            contador_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, Sizeof.cl_int * numGrupos, Pointer.to(inicio_contador), null);

            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(livro_buffer));
            clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(palavra_buffer));
            clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(contador_buffer));
            clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[]{bytes_livro.length}));
            clSetKernelArg(kernel, 4, Sizeof.cl_int, Pointer.to(new int[]{bytes_palavra.length}));

            long inicio = System.currentTimeMillis();
            clEnqueueNDRangeKernel(queue, kernel, 1, null, new long[]{globalSize}, new long[]{localSize}, 0, null, null);
            
            int[] resultados = new int[numGrupos];
            clEnqueueReadBuffer(queue, contador_buffer, true, 0, Sizeof.cl_int * numGrupos, Pointer.to(resultados), 0, null, null);
            
            int total = 0;
            for (int r : resultados) total += r;
            System.out.println("Ocorrências: " + total + " | Tempo: " + (System.currentTimeMillis() - inicio) + "ms");

        } finally {
            if (contador_buffer != null) clReleaseMemObject(contador_buffer);
            if (palavra_buffer != null) clReleaseMemObject(palavra_buffer);
            if (livro_buffer != null) clReleaseMemObject(livro_buffer);
            if (kernel != null) clReleaseKernel(kernel);
            if (program != null) clReleaseProgram(program);
            if (queue != null) clReleaseCommandQueue(queue);
            if (context != null) clReleaseContext(context);
        }
    }
}
