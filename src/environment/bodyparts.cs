using UnityEngine;
using System.Collections;
using System.IO;
using UnityEngine.UI;
using System;
using System.Text;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Collections.Generic;

// copied from https://stackoverflow.com/questions/42717713/unity-live-video-streaming
public class CameraPart : MonoBehaviour
{
    WebCamTexture webCam;
    public RawImage myImage;
    Texture2D currentTexture;

    private TcpListener listner;
    private const int port = 8010;
    private bool stop = false;

    private List<TcpClient> clients = new List<TcpClient>();

    private void Start()
    {
        // Open the Camera on the desired device, in my case IPAD pro
        webCam = new WebCamTexture();
        // Get all devices , front and back camera
        webCam.deviceName = WebCamTexture.devices[WebCamTexture.devices.Length - 1].name;

        // request the lowest width and heigh possible
        webCam.requestedHeight = 10;
        webCam.requestedWidth = 10;


        webCam.Play();

        /
        currentTexture = new Texture2D(webCam.width, webCam.height);

        // Connect to the server
        listner = new TcpListener(port);

        listner.Start();

        // Create Seperate thread for requesting from client
        Loom.RunAsync(() => {

            while (!stop)
            {
                // Wait for client approval
                var client = listner.AcceptTcpClient();
                // We are connected
                clients.Add(client);


                Loom.RunAsync(() =>
                {
                    while (!stop)
                    {

                        var stremReader = client.GetStream();

                        if (stremReader.CanRead)
                        {
                            // we need storage for data
                            using (var messageData = new MemoryStream())
                            {
                                Byte[] buffer = new Byte[client.ReceiveBufferSize];


                                while (stremReader.DataAvailable)
                                {
                                    int bytesRead = stremReader.Read(buffer, 0, buffer.Length);

                                    if (bytesRead == 0)
                                        break;

                                    // Writes to the data storage
                                    messageData.Write(buffer, 0, bytesRead);

                                }

                                if (messageData.Length > 0)
                                {
                                    // send pngImage
                                    SendPng(client);

                                }

                            }
                        }
                    }
                });
            }

        });



    }

    private void Update()
    {
        myImage.texture = webCam;
    }


    // Read video pixels and send them to the client
    private void SendPng (TcpClient client)
    {
        Loom.QueueOnMainThread(() =>
        {
            // Get the webcame texture pixels
            currentTexture.SetPixels(webCam.GetPixels());
            var pngBytes = currentTexture.EncodeToPNG();


            // Want to Write
            var stream = client.GetStream();

            // Write the image bytes
            stream.Write(pngBytes, 0, pngBytes.Length);

            // send it
            stream.Flush();

        });
    }

    // stop everything
    private void OnApplicationQuit()
    {
        webCam.Stop();
        stop = true;
        listner.Stop();

        foreach (TcpClient c in clients)
            c.Close();
    }



}